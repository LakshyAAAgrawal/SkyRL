from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.hover.score import metric_with_feedback
from typing import Dict, Any
from omegaconf import DictConfig


DEFAULT_SYSTEM_PROMPT = "Given a claim, and some documents related to the claim, write a query to retrieve documents supporting the claim. Respond with your reasoning and the query at the end formatted as '### reasoning: <your reasoning here>\n### query: <your query here>'"

class HoVerEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_model" in extras, "reward_model field is required"
        assert extras["reward_model"]["method"] == "rule"
        assert "claim" in extras["reward_model"]
        assert "supporting_facts" in extras["reward_model"]
        assert "label" in extras["reward_model"]
        self.extras = extras

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step

        score, feedback = metric_with_feedback(action, self.extras['reward_model'])
        reward = score
        metadata = {"score": score, "feedback": feedback}

        # No observation in aime, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
