from src.loader import load_documents

docs = load_documents()
for d in docs:
    print(d["source"], "-", len(d["text"]), "characters")