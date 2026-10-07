from src.loader import load_documents
from src.chunking import chunk_documents
from src.embeddings import embed_documents, embed_query
from src.stores import NumpyStore, FaissStore, ChromaStore
from src.rerank import rerank

STORES = {"numpy": NumpyStore, "faiss": FaissStore, "chroma": ChromaStore}


def prepare(strategy="recursive"):
    """Load, chunk and embed once; the result can fill any store."""
    chunks = chunk_documents(load_documents(), strategy=strategy)
    vectors = embed_documents([c["text"] for c in chunks])
    return chunks, vectors


def build_store(store_name, chunks, vectors):
    store = STORES[store_name]()
    store.add(chunks, vectors)
    return store


def retrieve(store, question, k=5, use_rerank=True):
    if use_rerank:
        candidates = store.search(embed_query(question), k=20)
        return rerank(question, candidates, top_n=k)
    return store.search(embed_query(question), k=k)