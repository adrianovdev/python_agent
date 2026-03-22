from pymilvus import connections, Collection
import numpy as np
import requests

# -----------------------------
# 1. Embedding model (replace with your own)
# -----------------------------
def embed_text(text: str) -> list:
    # Example: using a dummy embedding for demonstration
    # Replace with your real embedding model call
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(768).tolist()


# -----------------------------
# 2. LLM call (replace with your own)
# -----------------------------
def call_llm(prompt: str) -> str:
    # Replace with your LLM provider (OpenAI, Azure, local model, etc.)
    return f"(Stub LLM) Answer based on context:\n{prompt}"


# -----------------------------
# 3. Milvus Agent
# -----------------------------
class MilvusAgent:
    def __init__(self, host="localhost", port="19530", collection_name="documents"):
        connections.connect(alias="default", host=host, port=port)
        self.collection = Collection(collection_name)
        self.collection.load()

    def search(self, query: str, top_k: int = 5):
        query_vec = embed_text(query)

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[query_vec],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        hits = []
        for hit in results[0]:
            hits.append({
                "text": hit.entity.get("text"),
                "score": hit.distance
            })

        return hits

    def answer(self, query: str) -> str:
        docs = self.search(query, top_k=5)

        context = "\n".join(f"- {d['text']}" for d in docs)

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

        return call_llm(prompt)


# -----------------------------
# 4. Run the agent
# -----------------------------
if __name__ == "__main__":
    agent = MilvusAgent()
    question = "What does the system say about refund policies?"
    print(agent.answer(question))

