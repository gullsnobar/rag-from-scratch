from pathlib import Path
from pypdf import PdfReader

def load_documents(folder="data"):
    docs = []
    for path in Path(folder).iterdir():
        if path.suffix == ".pdf":
            text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        elif path.suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        else:
            continue
        docs.append({"text": text, "source": path.name})
    return docs