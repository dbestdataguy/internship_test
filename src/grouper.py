import json
import math
import os
from groq import Groq
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from src.embeddings import get_embeddings
from src.prompt_templates import build_labeling_prompt

load_dotenv()

def estimate_num_clusters(n: int) -> int:
    k = max(2, min(int(math.sqrt(n / 2)), n // 2))
    return k


def cluster_transactions(descriptions: list[str]) -> dict:
    embeddings = get_embeddings(descriptions)
    n = len(descriptions)
    k = estimate_num_clusters(n)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(descriptions[idx])

    return clusters


def parse_llm_response(raw_response: str) -> dict:
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw response:\n{cleaned}")


def label_clusters_with_llm(clusters: dict) -> dict:
    prompt = build_labeling_prompt(clusters)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    raw_response = response.choices[0].message.content
    parsed = parse_llm_response(raw_response)
    return parsed


def build_final_output(parsed_llm_output: dict, total_input: int) -> dict:
    groups = parsed_llm_output.get("groups", [])
    ungrouped = parsed_llm_output.get("ungrouped", [])

    output = {
        "groups": groups,
        "ungrouped": ungrouped,
        "summary": {
            "total_input": total_input,
            "total_groups": len(groups),
            "ungrouped_count": len(ungrouped)
        }
    }
    return output


def group_transactions(descriptions: list[str]) -> dict:
    if not descriptions:
        raise ValueError("Input list is empty.")

    clusters = cluster_transactions(descriptions)
    parsed_llm_output = label_clusters_with_llm(clusters)
    final_output = build_final_output(parsed_llm_output, len(descriptions))

    return final_output