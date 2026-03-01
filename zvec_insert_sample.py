
from zvec import Doc

def insert_json_doc_dual(collection, json_obj: dict):
    text = json_obj["text"]
    doc_id = json_obj["id"]

    openai_emb = embed(text)        # from previous OpenAI example
    local_emb = local_embed(text)   # from step 2

    doc = Doc(
        id=doc_id,
        vectors={
            "openai_embedding": openai_emb,
            "local_embedding": local_emb,
        },
        fields={
            "title": json_obj.get("title"),
            "text": text,
        },
    )
    return collection.insert(doc)