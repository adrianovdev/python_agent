# orders_api.py
# uvicorn orders_api:app --reload --host 0.0.0.0 --port 8000

# curl -X POST http://localhost:8000/orders/query \
#   -H "Content-Type: application/json" \
#   -d '{
#         "query": "How many high-value completed orders do I have and what is their total amount?",
#         "top_k": 20,
#         "expr": "amount_value > 40 and status == \"COMPLETED\""
#       }'


from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from orders_agent import PayPalOrdersAgent

app = FastAPI(title="PayPal Orders RAG API")
agent = PayPalOrdersAgent()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    expr: str | None = None  # optional Milvus filter expression

class OrderDoc(BaseModel):
    order_id: str
    status: str
    currency: str
    amount_value: float
    json_text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    documents: List[OrderDoc]

@app.post("/orders/query", response_model=QueryResponse)
def query_orders(req: QueryRequest):
    result = agent.answer(req.query, top_k=req.top_k, expr=req.expr)
    docs = [OrderDoc(**d) for d in result["documents"]]
    return QueryResponse(answer=result["answer"], documents=docs)

@app.get("/health")
def health():
    return {"status": "ok"}

