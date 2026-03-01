"""Small demo showing how to use `zvec_service` helpers.

This file is safe to run even if `zvec` or `sentence-transformers` are not installed
because the service functions are defensive about imports. Replace the placeholder
client with your actual zvec client instance.
"""
from zvec_service import local_embed, build_collection_schema, create_collection, insert_documents


def demo_local_embedding():
    text = "This is a short demo text for embeddings."
    emb = local_embed(text)
    print(f"Embedding length: {len(emb)}")


def demo_build_schema():
    fields = [{"name": "title"}, {"name": "text"}]
    vectors = [{"name": "local_embedding", "dimension": 384, "metric": "COSINE"}]
    schema = build_collection_schema("demo_documents", fields, vectors)
    print("Built schema:", schema)


if __name__ == "__main__":
    demo_local_embedding()
    demo_build_schema()
    print("Demo finished. Replace demo code with your zvec client calls to create collections and insert docs.")
