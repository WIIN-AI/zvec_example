from sentence_transformers import SentenceTransformer

LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
local_model = SentenceTransformer(LOCAL_MODEL_NAME)  # loads from HF
LOCAL_DIM = 384  # model’s embedding size


def local_embed(text: str) -> list[float]:
    # SentenceTransformers handles batching; here we pass a single string
    emb = local_model.encode(
        [text],
        normalize_embeddings=True,  # unit-length vectors
        convert_to_numpy=True,
    )[0]
    return emb.tolist()