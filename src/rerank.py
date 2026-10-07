from sentence_transformers import CrossEncoder

_reranker = None


def rerank(question, results, top_n=5):
    """Re-score (chunk, score) candidates by how well each answers the question."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = _reranker.predict([(question, chunk["text"]) for chunk, _ in results])
    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, float(s)) for (chunk, _), s in ranked[:top_n]]