# ingest_orders.py
import os
import json
import hashlib
from typing import List, Dict, Any

from pymilvus import connections, Collection
from setup_milvus_orders import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, EMBEDDING_DIM

import numpy as np

def embed_text(text: str) -> list:
    # Replace with real embedding model
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(EMBEDDING_DIM).tolist()

def hash_token(value: str, length: int = 8) -> str:
    h = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return h[:length]

def anonymize_order(order: Dict[str, Any]) -> Dict[str, Any]:
    order = dict(order)  # shallow copy

    payer = order.get("payer", {})
    email = payer.get("email_address")
    name = payer.get("name", {})

    if email:
        token = hash_token(email)
        domain = email.split("@")[-1] if "@" in email else "example.com"
        payer["email_address"] = f"anon+{token}@{domain}"

    if name:
        name["given_name"] = "ANON"
        name["surname"] = "ANON"
        payer["name"] = name

    if "payer_id" in payer:
        payer["payer_id"] = "ANON-PAYER"

    order["payer"] = payer
    return order

def extract_fields(order: Dict[str, Any]):
    order_id = order.get("id", "")
    status = order.get("status", "")
    amount = order.get("purchase_units", [{}])[0].get("amount", {}) \
             or order.get("amount", {})
    value = float(amount.get("value", 0.0))
    currency = amount.get("currency_code") or amount.get("currency") or ""

    return order_id, status, currency, value

def load_orders_from_dir(path: str) -> List[Dict[str, Any]]:
    orders = []
    for fname in os.listdir(path):
        if not fname.lower().endswith(".json"):
            continue
        full = os.path.join(path, fname)
        with open(full, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                orders.append(data)
            except Exception:
                continue
    return orders

def ingest(path: str = "./paypal_orders"):
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)

    raw_orders = load_orders_from_dir(path)

    order_ids, statuses, currencies, amounts, json_texts, embeddings = [], [], [], [], [], []

    for raw in raw_orders:
        anon = anonymize_order(raw)
        order_id, status, currency, value = extract_fields(anon)
        text_repr = json.dumps(anon, ensure_ascii=False)

        order_ids.append(order_id)
        statuses.append(status)
        currencies.append(currency)
        amounts.append(value)
        json_texts.append(text_repr)
        embeddings.append(embed_text(text_repr))

    entities = [order_ids, statuses, currencies, amounts, json_texts, embeddings]
    collection.insert(entities)
    collection.flush()
    print(f"Ingested {len(order_ids)} orders.")

if __name__ == "__main__":
    ingest()
