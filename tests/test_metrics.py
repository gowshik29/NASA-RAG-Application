from evaluation.metrics import recall_at_k, reciprocal_rank, numeric_overlap

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b", "c"], 3) == 1.0

def test_reciprocal_rank():
    assert reciprocal_rank(["x", "b"], ["b"]) == 0.5

def test_numeric_overlap():
    assert numeric_overlap("1,000 to 1,500 km/s", ["1,000", "1,500"]) == 1.0
