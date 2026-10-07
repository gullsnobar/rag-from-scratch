import sys
from src.rag import STORES, prepare, build_store, retrieve

# Usage: python main.py [numpy|faiss|chroma] [recursive|fixed]
store_name = sys.argv[1] if len(sys.argv) > 1 else "numpy"
strategy = sys.argv[2] if len(sys.argv) > 2 else "recursive"

if store_name not in STORES:
    print(f"Unknown store '{store_name}'. Choose from: {', '.join(STORES)}")
    sys.exit(1)

chunks, vectors = prepare(strategy)
store = build_store(store_name, chunks, vectors)
print(f"\nIndexed {len(chunks)} chunks in {store_name} using {strategy} chunking.\n")

while True:
    question = input("Ask a question (or 'quit'): ").strip()
    if question.lower() == "quit":
        break
    if not question:
        continue
    for chunk, score in retrieve(store, question):
        print(f"\n[{score:.2f}] {chunk['chunk_id']}")
        print(chunk["text"][:300])
    print("\n" + "-" * 60)