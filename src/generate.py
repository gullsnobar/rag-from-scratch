import ollama

MODEL = "llama3.2"


def build_prompt(question, results):
    context = "\n\n".join(
        f"[{i}] (source: {chunk['source']})\n{chunk['text']}"
        for i, (chunk, _) in enumerate(results, 1)
    )
    return f"""Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I couldn't find that in the documents."
Cite the context numbers you used, like [1] or [2].

Context:
{context}

Question: {question}
Answer:"""


def generate_answer(question, results):
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(question, results)}],
        options={"temperature": 0},
    )
    return response["message"]["content"]