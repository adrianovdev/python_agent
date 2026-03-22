# setup_milvus_orders.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "paypal_orders"
EMBEDDING_DIM = 768

def create_collection():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="order_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="currency", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="amount_value", dtype=DataType.FLOAT),
        FieldSchema(name="json_text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(fields, description="PayPal orders (anonymized)")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Collection and index created.")

if __name__ == "__main__":
    create_collection()
