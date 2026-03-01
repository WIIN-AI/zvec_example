
EMBED_DIM_OPENAI = 1536      # e.g., text-embedding-3-small
EMBED_DIM_LOCAL = 384        # all-MiniLM-L6-v2

schema = CollectionSchema(
    name="documents_dual",
    fields=[
        FieldSchema("title", DataType.STRING),
        FieldSchema("text", DataType.STRING),
    ],
    vectors=[
        VectorSchema(
            name="openai_embedding",
            data_type=DataType.VECTOR_FP32,
            dimension=EMBED_DIM_OPENAI,
            index_param=HnswIndexParam(metric_type=IndexMetricType.COSINE),
        ),
        VectorSchema(
            name="local_embedding",
            data_type=DataType.VECTOR_FP32,
            dimension=EMBED_DIM_LOCAL,
            index_param=HnswIndexParam(metric_type=IndexMetricType.COSINE),
        ),
    ],
)

