from zvec import CollectionSchema, FieldSchema, VectorSchema, DataType
from zvec import HnswIndexParam, IndexMetricType

schema = CollectionSchema(
    name="documents",
    fields=[
        FieldSchema("title", DataType.STRING),
        FieldSchema("text", DataType.STRING),
    ],
    vectors=[
        VectorSchema(
            name="local_embedding",
            data_type=DataType.VECTOR_FP32,
            dimension=LOCAL_DIM,  # 384 for all-MiniLM-L6-v2
            index_param=HnswIndexParam(metric_type=IndexMetricType.COSINE),
        )
    ],
)



