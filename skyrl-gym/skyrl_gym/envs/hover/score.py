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

def metric_with_feedback(gen_str, extras):
    example = extras

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