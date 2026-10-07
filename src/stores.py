import numpy as np
import faiss
import chromadb

class NumpyStore:
    """Vector store built from scratch: brute-force cosine similarity."""

    def __init__(self):
        self.vectors = None
        self.chunks = []

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec, k=3):
        scores = self.vectors @ query_vec          # similarity to every chunk
        top = np.argsort(scores)[::-1][:k]         # indices of the k highest
        return [(self.chunks[i], float(scores[i])) for i in top]

class FaissStore:
    """FAISS: fast similarity-search library. Stores only vectors."""

    def __init__(self, dim=384):
        self.index = faiss.IndexFlatIP(dim)   # exact inner-product search
        self.chunks = []                      # we must track chunk text ourselves

    def add(self, chunks, vectors):
        self.index.add(np.asarray(vectors, dtype="float32"))
        self.chunks.extend(chunks)

    def search(self, query_vec, k=3):
        scores, ids = self.index.search(np.asarray([query_vec], dtype="float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


class ChromaStore:
    """ChromaDB: full vector database. Stores vectors, text, and metadata on disk."""

    def __init__(self, path="chroma_db", name="docs"):
        client = chromadb.PersistentClient(path=path)
        try:
            client.delete_collection(name)    # start fresh each run
        except Exception:
            pass
        self.col = client.create_collection(name)

    def add(self, chunks, vectors):
        self.col.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
        )

    def search(self, query_vec, k=3):
        r = self.col.query(query_embeddings=[query_vec.tolist()], n_results=k)
        results = []
        for text, meta, cid, dist in zip(r["documents"][0], r["metadatas"][0],
                                         r["ids"][0], r["distances"][0]):
            chunk = {"text": text, "source": meta["source"], "chunk_id": cid}
            results.append((chunk, 1 - dist / 2))   # squared L2 -> cosine (unit vectors)
        return results        