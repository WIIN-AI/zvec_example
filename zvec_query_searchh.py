from zvec import VectorQuery

def search_local(collection, query_text: str, topk: int = 5):
    q_vec = local_embed(query_text)
    return collection.query(
        vectors=VectorQuery("local_embedding", vector=q_vec),
        topk=topk,
        output_fields=["title", "text"],
    )

def search_openai(collection, query_text: str, topk: int = 5):
    q_vec = embed(query_text)  # OpenAI
    return collection.query(
        vectors=VectorQuery("openai_embedding", vector=q_vec),
        topk=topk,
        output_fields=["title", "text"],
    )


def insert_json_doc(collection, json_obj: dict):
    doc_id = json_obj["id"]
    text = json_obj["text"]
    emb = embed(text)

    doc = Doc(
        id=doc_id,
        vectors={"embedding": emb},
        fields={
            "title": json_obj.get("title"),
            "text": text,
        },
    )
    return collection.insert(doc)


def search(collection, query_text: str, topk: int = 5):
    q_emb = embed(query_text)
    results = collection.query(
        vectors=VectorQuery("embedding", vector=q_emb),
        topk=topk,
        filter=None,
        include_vector=False,
        output_fields=["title", "text"],
    )
    return results
