from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_documents(texts):
    return get_model().encode(texts, normalize_embeddings=True,
                              batch_size=32, show_progress_bar=True)


def embed_query(text):
    # BGE models work best when questions get this prefix and documents don't
    return get_model().encode([QUERY_PREFIX + text], normalize_embeddings=True)[0]