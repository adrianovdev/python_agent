# config.py
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "documents"

EMBEDDING_DIM = 768
EMBEDDING_METRIC = "IP"  # or "L2"

# LLM / embedding provider config (env vars in real life)
OPENAI_API_KEY = "YOUR_KEY_HERE"
