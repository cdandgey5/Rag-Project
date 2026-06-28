from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import os
from pathlib import Path

# Set Ollama models
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)
Settings.llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",
    request_timeout=300.0
)

documents = []

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def read_html(path):
    from bs4 import BeautifulSoup
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def read_rtf(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # Strip RTF control words and braces
    import re
    content = re.sub(r'\{\*?\\[^{}]+}|[{}]|\\[a-z]+\d* ?', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    return content.strip()

# File type map
SUPPORTED = {
    ".txt":  read_txt,
    ".pdf":  read_pdf,
    ".html": read_html,
    ".htm":  read_html,
    ".rtf":  read_rtf,
}

# Walk ALL subfolders recursively
docs_path = Path("./documents")
print("📖 Reading documents...\n")

# rglob("*") finds files in all subfolders at any depth
for file in sorted(docs_path.rglob("*")):
    if file.is_dir():
        continue  # Skip folders, only process files

    ext = file.suffix.lower()
    relative = file.relative_to(docs_path)  # e.g. "june28/meeting-notes.txt"

    if ext not in SUPPORTED:
        print(f"  Skipping: {relative} (unsupported type)")
        continue

    try:
        text = SUPPORTED[ext](file)

        if not text.strip():
            print(f"  ⚠️  Empty: {relative}")
            continue

        documents.append(Document(
            text=text,
            metadata={
                "filename": file.name,
                "folder": str(file.parent.relative_to(docs_path)),
                "filepath": str(relative)
            }
        ))
        print(f"  ✓ {relative} ({len(text)} characters)")

    except Exception as e:
        print(f"  ❌ Failed: {relative} → {e}")

print(f"\n✓ Loaded {len(documents)} documents\n")

# Create index
print("⚙️  Creating embeddings...")
index = VectorStoreIndex.from_documents(documents)

# Save index
print("💾 Saving index...")
index.storage_context.persist(persist_dir="./index")

print("\n✅ Setup complete!")
print("💡 Run: python chat.py")
