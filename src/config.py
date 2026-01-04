import os

APP_MODE = os.getenv("APP_MODE", "demo").lower()

if APP_MODE not in {"demo", "custom"}:
    raise ValueError("APP_MODE must be 'demo' or 'custom'")


FINGERPRINT = {
    "dataset": "demo_movies_100" if APP_MODE == "demo" else "custom_dataset",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "chunking_version": "v1.0",
    "chunk_size": 500,
    "chunk_overlap": 100,
}
