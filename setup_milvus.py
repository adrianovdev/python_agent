# setup_milvus.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from config import MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, EMBEDDING_DIM, EMBEDDING_METRIC

def create_collection():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]

    schema = CollectionSchema(fields, description="RAG documents")
    collection = Collection(name=MILVUS_COLLECTION, schema=schema)

    index_params = {
        "metric_type": EMBEDDING_METRIC,
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Collection and index created.")

if __name__ == "__main__":
    create_collection()
