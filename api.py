# This file defines the FastAPI application for the Milvus RAG service.
# uvicorn api:app --reload --host 0.0.0.0 --port 8000

# curl -X POST http://localhost:8000/query \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is our refund policy?", "top_k": 5}'

# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from agent.py import MilvusRAGAgent

app = FastAPI(title="Milvus RAG Service")
agent = MilvusRAGAgent()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    documents: list

@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    result = agent.answer(req.query, top_k=req.top_k)
    return QueryResponse(**result)

@app.get("/health")
def health():
    return {"status": "ok"}
