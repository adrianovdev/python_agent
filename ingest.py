# ingest.py
import os
from typing import List
from pymilvus import connections, Collection
from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION
from models import embed_text

DATA_DIR = "./data"

def load_text_files(path: str) -> List[str]:
    docs = []
    for fname in os.listdir(path):
        full = os.path.join(path, fname)
        if os.path.isfile(full) and fname.lower().endswith((".txt", ".md")):
            with open(full, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def simple_chunk(text: str, max_chars: int = 800) -> List[str]:
    chunks, buf = [], []
    length = 0
    for line in text.splitlines():
        if length + len(line) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, length = [], 0
        buf.append(line)
        length += len(line)
    if buf:
        chunks.append("\n".join(buf))
    return chunks

def ingest():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(MILVUS_COLLECTION)

    raw_docs = load_text_files(DATA_DIR)
    texts, embeddings = [], []

    for doc in raw_docs:
        for chunk in simple_chunk(doc):
            texts.append(chunk)
            embeddings.append(embed_text(chunk))

    entities = [texts, embeddings]
    collection.insert(entities)
    collection.flush()
    print(f"Ingested {len(texts)} chunks.")

if __name__ == "__main__":
    ingest()
    