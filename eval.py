from src.rag import STORES, prepare, build_store, retrieve

TEST_SET = [
    ("What year was the Transformer architecture introduced?", "2017"),
    ("What are the main stages of training an LLM?", "RLHF"),
    ("How does attention decide which tokens matter?", "assigns weights"),
    ("Why do LLMs use pre-layer normalization?", "training stability"),
    ("What share of problems did Codex solve with repeated sampling?", "77.5"),
    ("What training objective does UL2 use?", "mixture of denoisers"),
    ("How much faster did PaLM's parallel layers make training?", "factor of 15"),
    ("How are words turned into numbers for the neural network?", "Token ID"),
    ("What was UK graduate unemployment in 2025?", "5.3"),
]


def normalize(text):
    return " ".join(text.split()).lower()


def hit(results, expected):
    return any(normalize(expected) in normalize(chunk["text"]) for chunk, _ in results)


rows = []
for strategy in ("fixed", "recursive"):
    chunks, vectors = prepare(strategy)
    for store_name in STORES:
        store = build_store(store_name, chunks, vectors)
        for use_rerank in (False, True):
            misses = [q for q, exp in TEST_SET
                      if not hit(retrieve(store, q, k=5, use_rerank=use_rerank), exp)]
            score = len(TEST_SET) - len(misses)
            rows.append((strategy, store_name, "yes" if use_rerank else "no", score, misses))

print("\n| Chunking | Store | Rerank | Hits (top 5) |")
print("|---|---|---|---|")
for strategy, store_name, rr, score, _ in rows:
    print(f"| {strategy} | {store_name} | {rr} | {score}/{len(TEST_SET)} |")

print("\nMissed questions per configuration:")
for strategy, store_name, rr, score, misses in rows:
    if misses:
        print(f"\n{strategy} / {store_name} / rerank={rr}:")
        for q in misses:
            print("  -", q)