def fixed_size_chunks(text, size=800, overlap=100):
    """Strategy 1: cut every `size` characters, overlapping by `overlap`."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def recursive_chunks(text, max_size=800, separators=("\n\n", "\n", ". ", " ")):
    """Strategy 2: split on paragraphs, then lines, sentences, words."""
    if len(text) <= max_size:
        return [text.strip()] if text.strip() else []
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        chunks, current = [], ""
        for part in text.split(sep):
            piece = part + sep
            if len(current) + len(piece) <= max_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                if len(piece) > max_size:
                    chunks.extend(recursive_chunks(piece, max_size, separators[i + 1:]))
                    current = ""
                else:
                    current = piece
        if current.strip():
            chunks.append(current.strip())
        return chunks
    return fixed_size_chunks(text, max_size, 0)


def chunk_documents(docs, strategy="recursive", size=800):
    fn = recursive_chunks if strategy == "recursive" else fixed_size_chunks
    out = []
    for d in docs:
        for i, c in enumerate(fn(d["text"], size)):
            out.append({"text": c, "source": d["source"], "chunk_id": f"{d['source']}-{i}"})
    return out