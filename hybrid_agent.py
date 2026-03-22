# hybrid_agent.py
from query_classifier import classify_query
from sql_backend import SQLBackend
from milvus_backend import MilvusBackend
from llm_openai import call_llm

class HybridAgent:
    def __init__(self):
        self.sql = SQLBackend()
        self.milvus = MilvusBackend()

    def answer(self, query: str):
        query_type = classify_query(query)

        if query_type == "aggregation":
            # Route to SQL
            result = self.sql.run_custom_aggregation(query)
            prompt = f"""
You are a helpful analyst. The user asked:

"{query}"

Here is the aggregated data from the SQL backend:
{result}

Explain the result clearly and concisely.
"""
            return call_llm(prompt)

        else:
            # Route to Milvus
            docs = self.milvus.semantic_search(query)
            context = "\n\n".join(d["json_text"] for d in docs)
            prompt = f"""
You are a helpful analyst. Use ONLY the context to answer.

Context:
{context}

Question:
{query}

Answer:
"""
            return call_llm(prompt)
