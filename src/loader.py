import logging
import re
from pathlib import Path
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)   # hides "Overwriting cache" noise


def drop_references(text):
    """Cut the reference list at the end of academic papers."""
    matches = list(re.finditer(r"\n\s*(References|Bibliography)\s*\n", text, re.IGNORECASE))
    if matches and matches[-1].start() > len(text) * 0.5:
        return text[:matches[-1].start()]
    return text


def clean_text(text):
    text = re.sub(r"-\n(\w)", r"\1", text)     # rejoin hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)        # collapse repeated spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse big gaps
    return text.strip()


def load_documents(folder="data"):
    docs = []
    for path in sorted(Path(folder).iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        text = clean_text(drop_references(text))
        if len(text) < 200:
            print(f"Skipping {path.name}: almost no extractable text")
            continue
        docs.append({"text": text, "source": path.name})
    return docs