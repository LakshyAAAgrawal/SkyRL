from typing import Any, Callable, TypedDict

from gepa.api import optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
import os
import time
import requests
from typing import List, Dict
import litellm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import re
import string
import unicodedata
from collections import Counter
from functools import partial
from gepa_artifact.gepa_artifact.benchmarks.hover.hover_program import search

retrieve_k = partial(search, k=7)
retrieve_10 = partial(search, k=10)

def normalize_text(s):
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_query(gen_str):
    if '### query: ' not in gen_str:
        return None, "'### query: ' not found in your response. You must include the query to be executed in the format '### query: <your query here>'"
    
    query = gen_str.split('### query: ')[1].strip()
    return query, None

def metric_with_feedback(data, gen_str):
    example = data

    extracted_query, feedback = extract_query(gen_str)
    if extracted_query is None:
        return 0, feedback
    
    hop1_retrieved_docs = example['hop1_retrieved_docs']
    hop2_retrieved_docs = retrieve_10(extracted_query).passages
    retrieved_docs = hop1_retrieved_docs + hop2_retrieved_docs

    gold_titles = set(
        map(
            normalize_text,
            [doc["key"] for doc in example["supporting_facts"]],
        )
    )
    found_titles = set(
        map(
            normalize_text,
            [c.split(" | ")[0] for c in retrieved_docs],
        )
    )

    score = 1 if gold_titles.issubset(found_titles) else 0

    gold_titles_found_in_pred = gold_titles.intersection(found_titles)
    gold_titles_not_found_in_pred = gold_titles.difference(found_titles)

    if gold_titles_found_in_pred and gold_titles_not_found_in_pred:
        feedback_text = f"Your queries correctly retrieved the following relevant evidence documents: {gold_titles_found_in_pred}, but missed the following relevant evidence documents: {gold_titles_not_found_in_pred}. Try and think about how you can rephrase or adjust your query to better target these?"
    elif gold_titles_found_in_pred:
        feedback_text = f"Your queries correctly retrieved all the relevant evidence documents!"
    elif gold_titles_not_found_in_pred:
        feedback_text = f"Your queries missed the following relevant evidence documents: {gold_titles_not_found_in_pred}. Try and think about how you can rephrase or adjust your query to better target these?"
    else:
        feedback_text = "Your queries did not retrieve any relevant evidence documents. Try and think about how you can rephrase or adjust your query to better target these?"


    return score, feedback_text


def vllm_model_callable(
    messages: List[Dict[str, str]],
) -> str:
    """Call a local vLLM OpenAI-compatible server and return the assistant message content.

    Expects `messages` in the same format used by the default LiteLLM path:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    response = litellm.completion(
        model="hosted_vllm/Qwen/Qwen3-8B",                   # For example: "hosted_vllm/meta-llama/Meta-Llama-3-8B-Instruct"
        messages=messages,
        api_base="http://localhost:8000/v1"                 # vLLM's OpenAI-compatible endpoint
    )
    return response["choices"][0]["message"]["content"].strip()


from typing import Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


# DataInst, Trajectory, RolloutOutput
class HoverDataInst(TypedDict):
    claim: str
    documents: list[str]

class DefaultTrajectory(TypedDict):
    data: HoverDataInst
    full_assistant_response: str

class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str


from gepa_artifact.gepa_artifact.benchmarks.hover import benchmark as hover_metas
hoverbench = hover_metas[0].benchmark()
from gepa_artifact.gepa_artifact.benchmarks.hover.hover_program import search

from functools import partial
retrieve_k = partial(search, k=7)
retrieve_10 = partial(search, k=10)

def format_user_message_for_hover(datapoint):
    retrieved_docs_hop1 = retrieve_k(datapoint['claim'])['passages']
    return f"""Claim: {datapoint['claim']}
Documents: {str(retrieved_docs_hop1)}
"""

from skyrl_gym.envs.hover.env import DEFAULT_SYSTEM_PROMPT

def create_messages(datapoint, system_prompt=DEFAULT_SYSTEM_PROMPT):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": format_user_message_for_hover(datapoint)
        }
    ]
    return messages


class HoverAdapter(GEPAAdapter[HoverDataInst, DefaultTrajectory, DefaultRolloutOutput]):
    def __init__(
        self,
        model: str | Callable,
        failure_score: float = 0.0,
        max_litellm_workers: int = 100,
        metric: Callable = None,
    ):
        if isinstance(model, str):
            import litellm
            self.litellm = litellm
        self.model = model

        self.failure_score = failure_score
        self.max_litellm_workers = max_litellm_workers
        self.metric = metric

    def evaluate(
        self,
        batch: list[HoverDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput]:
        start_time = time.time()
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

        system_content = next(iter(candidate.values()))

        litellm_requests = []

        for data in tqdm(batch):


            messages = create_messages(data, system_content)

            litellm_requests.append(messages)

        try:
            if isinstance(self.model, str):
                responses = [resp.choices[0].message.content.strip() for resp in self.litellm.batch_completion(model=self.model, messages=litellm_requests, max_workers=self.max_litellm_workers)]
            else:
                with ThreadPoolExecutor(max_workers=min(self.max_litellm_workers, len(litellm_requests))) as executor:
                    responses = list(
                        tqdm(
                            executor.map(self.model, litellm_requests),
                            total=len(litellm_requests),
                            desc="Local model",
                        )
                    )
        except Exception as e:
            raise e

        filtered_responses = []
        for response in responses:
            # remove all <think> and </think>
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            response = response.strip()
            filtered_responses.append(response)

        for data, assistant_response, messages in zip(batch, filtered_responses, litellm_requests):
            output = {"full_assistant_response": assistant_response}

            score, feedback = self.metric(data, assistant_response)

            outputs.append(output)
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                        "feedback": feedback,
                        "input": messages[-1]["content"],
                    }
                )

        elapsed_s = time.time() - start_time
        print(f"HoverAdapter.evaluate elapsed: {elapsed_s:.2f}s (batch={len(batch)})")
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[DefaultTrajectory, DefaultRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        ret_d: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        items: list[dict[str, Any]] = []
        trace_instances = list(zip(eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False))

        for trace_instance in trace_instances:
            traj, score, _ = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            feedback = traj["feedback"]
            input = traj["input"]
            d = {
                "Inputs": input,
                "Generated Outputs": generated_outputs,
                "Feedback": feedback,
            }

            items.append(d)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d



hover_adapter = HoverAdapter(
    model=vllm_model_callable,
    metric=metric_with_feedback,
)


for dset in [hoverbench.train_set, hoverbench.val_set, hoverbench.test_set]:
    for ex in dset:
        retrieved_docs_hop1 = retrieve_k(ex['claim'])['passages']
        ex['hop1_retrieved_docs'] = retrieved_docs_hop1

trainset, valset, testset = hoverbench.train_set, hoverbench.val_set, hoverbench.test_set


reflection_lm_vllm = (
    lambda prompt: litellm.completion(model="hosted_vllm/Qwen/Qwen3-8B", messages=[{"role": "user", "content": prompt}], api_base="http://localhost:8000/v1" )
    .choices[0]
    .message.content
)

# testset_results_after_opt = hover_adapter.evaluate(
#     testset,
#     {"instruction_prompt": DEFAULT_SYSTEM_PROMPT},
#     capture_traces=True,
# )


optimized_results = optimize(
    seed_candidate={"instruction_prompt": DEFAULT_SYSTEM_PROMPT},
    trainset=trainset,
    valset=valset,
    testset=testset,
    adapter=hover_adapter,
    reflection_lm=reflection_lm_vllm,
    use_wandb=True,
    wandb_api_key="a03f758538f9d6256812fa45e0440cc390015cc2",
    max_metric_calls=7000,
    reflection_minibatch_size=3,
    perfect_score=1,
    skip_perfect_score=False,
    run_dir="gepa_hover_300_7000",
    display_progress_bar=True,
)


testset_results_after_opt = hover_adapter.evaluate(
    testset,
    {"instruction_prompt": optimized_results.best_candidate["instruction_prompt"]},
    capture_traces=True,
)

