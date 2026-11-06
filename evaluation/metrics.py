import re

def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant) if relevant else 0.0

def reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for rank, item in enumerate(retrieved_ids, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def citation_precision(answer):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(bool(re.search(r"\[Source:.*?p\.\s*\d+.*?\]", s, re.I)) for s in sentences)
    return cited / len(sentences)

def numeric_overlap(answer, expected_numbers):
    return sum(n in answer for n in expected_numbers) / max(1, len(expected_numbers))
