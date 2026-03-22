# orders_agent.py
from typing import List, Dict, Any
from pymilvus import connections, Collection

from setup_milvus_orders import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME
from ingest_orders import EMBEDDING_DIM, embed_text  # or move embed_text to a shared module

def call_llm(prompt: str) -> str:
    # Replace with real LLM call
    return f"(Stub LLM)\n{prompt[:800]}"

class PayPalOrdersAgent:
    def __init__(self):
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()

    def search(
        self,
        query: str,
        top_k: int = 10,
        expr: str | None = None
    ) -> List[Dict[str, Any]]:
        qvec = embed_text(query)
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }

        results = self.collection.search(
            data=[qvec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["order_id", "status", "currency", "amount_value", "json_text"],
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "order_id": hit.entity.get("order_id"),
                "status": hit.entity.get("status"),
                "currency": hit.entity.get("currency"),
                "amount_value": float(hit.entity.get("amount_value")),
                "json_text": hit.entity.get("json_text"),
                "score": float(hit.distance),
            })
        return hits

    def build_prompt(self, query: str, docs: List[Dict[str, Any]]) -> str:
        context = "\n\n".join(
            f"Order ID: {d['order_id']}\nStatus: {d['status']}\nCurrency: {d['currency']}\nAmount: {d['amount_value']}\nData: {d['json_text']}"
            for d in docs
        )
        return f"""You are an analyst over anonymized PayPal order data.
Use ONLY the context to answer the question. Do not invent orders.

Context:
{context}

Question:
{query}

Answer with a clear, concise explanation and, if relevant, counts or summaries.
"""

    def answer(
        self,
        query: str,
        top_k: int = 10,
        expr: str | None = None
    ) -> Dict[str, Any]:
        docs = self.search(query, top_k=top_k, expr=expr)
        prompt = self.build_prompt(query, docs)
        answer = call_llm(prompt)
        return {"answer": answer, "documents": docs}
