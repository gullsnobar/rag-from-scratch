from src.loader import load_documents
from src.chunking import chunk_documents

docs = load_documents()

for strategy in ("fixed", "recursive"):
    chunks = chunk_documents(docs, strategy=strategy)
    print(f"\n=== {strategy}: {len(chunks)} chunks ===")
    print("Example chunk:\n", chunks[5]["text"][:400])