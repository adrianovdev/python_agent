# agent.py
from typing import List, Dict, Any
from pymilvus import connections, Collection
from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, EMBEDDING_METRIC
from models import embed_text, call_llm

class MilvusRAGAgent:
    def __init__(self):
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = Collection(MILVUS_COLLECTION)
        self.collection.load()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qvec = embed_text(query)
        search_params = {
            "metric_type": EMBEDDING_METRIC,
            "params": {"nprobe": 10},
        }

        results = self.collection.search(
            data=[qvec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "score": float(hit.distance),
            })
        return hits

    def build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(f"- {d['text']}" for d in docs)
        return f"""You are a helpful assistant. Use ONLY the context to answer.

Context:
{context}

Question:
{query}

Answer:"""

    def answer(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        docs = self.search(query, top_k=top_k)
        prompt = self.build_prompt(query, docs)
        answer = call_llm(prompt)
        return {"answer": answer, "documents": docs}
