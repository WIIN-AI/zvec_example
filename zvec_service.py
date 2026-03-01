"""
zvec_service.py

Lightweight wrapper helpers to work with Zvec collections and local embeddings.
Functions are defensive: they lazily import external dependencies and give helpful
errors when the Zvec client API isn't present in the running environment.

Main utilities:
- local embedding with SentenceTransformers (lazy-loaded)
- helpers to build CollectionSchema objects
- create / list / get collection wrappers that try common zvec APIs
- insert_documents that accepts a collection object or collection name

This file avoids forcing imports at module import time so it can be imported
in environments where dependencies aren't installed (useful for tests).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

_local_model = None
_local_lock = threading.Lock()
DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_sentence_transformers(model_name: str = DEFAULT_LOCAL_MODEL):
    """Lazily load the SentenceTransformer model and return it.

    Raises ImportError with guidance if the package isn't installed.
    """
    global _local_model
    if _local_model is None:
        with _local_lock:
            if _local_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except Exception as e:  # pragma: no cover - environment dependent
                    raise ImportError(
                        "Please install sentence-transformers (pip install sentence-transformers) "
                        "to use local embeddings: "
                        f"original error: {e}"
                    )
                _local_model = SentenceTransformer(model_name)
    return _local_model


def local_embed(text: str, model_name: str = DEFAULT_LOCAL_MODEL) -> List[float]:
    """Return a single embedding vector for `text` as a python list of floats.

    This uses SentenceTransformers under the hood and normalizes embeddings.
    """
    model = _load_sentence_transformers(model_name)
    emb = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return emb.tolist()


# Helper construction utilities for CollectionSchema and related types


def build_collection_schema(
    name: str,
    fields: List[Dict[str, Any]],
    vectors: List[Dict[str, Any]],
    zvec_module: Optional[Any] = None,
):
    """Return a zvec.CollectionSchema constructed from simple dict descriptors.

    fields: list of {"name": str, "type": DataType.<X>} (or strings for name only)
    vectors: list of {"name": str, "dimension": int, "metric": "COSINE"|"L2"}

    If zvec_module is not provided, this function will import `zvec` lazily.
    """
    if zvec_module is None:
        try:
            import zvec as zvec_module  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "zvec package is required to build a CollectionSchema. "
                f"Install it or pass a zvec module object: {e}"
            )

    # Map simple field descriptors to FieldSchema objects
    field_objs = []
    for f in fields:
        if isinstance(f, str):
            field_objs.append(zvec_module.FieldSchema(f, zvec_module.DataType.STRING))
        elif isinstance(f, dict):
            name = f["name"]
            dtype = f.get("dtype", zvec_module.DataType.STRING)
            field_objs.append(zvec_module.FieldSchema(name, dtype))
        else:
            raise ValueError("fields must be str or dict descriptors")

    vector_objs = []
    for v in vectors:
        vname = v["name"]
        dim = int(v["dimension"])
        metric = v.get("metric", "COSINE")
        metric_type = getattr(zvec_module.IndexMetricType, metric, None)
        if metric_type is None:
            # fallback to COSINE
            metric_type = zvec_module.IndexMetricType.COSINE
        index_param = zvec_module.HnswIndexParam(metric_type=metric_type)
        vector_objs.append(
            zvec_module.VectorSchema(
                name=vname,
                data_type=zvec_module.DataType.VECTOR_FP32,
                dimension=dim,
                index_param=index_param,
            )
        )

    return zvec_module.CollectionSchema(name=name, fields=field_objs, vectors=vector_objs)


# Wrapper utilities that attempt multiple common Zvec client APIs so the
# service can work with different setups.


def _get_zvec_module():
    try:
        import zvec  # type: ignore
        return zvec
    except Exception as e:  # pragma: no cover
        raise ImportError("zvec package is required; install it (pip install zvec) or provide a client")


def create_collection(
    client: Optional[Any],
    name: str,
    fields: List[Dict[str, Any]],
    vectors: List[Dict[str, Any]],
    *,
    zvec_module: Optional[Any] = None,
):
    """Create a collection using a provided client or the zvec module.

    The function will try the following in order:
    - if `client` has a `create_collection(schema)` method, call it
    - if `zvec` module has `create_collection(schema)` call that

    Returns the created collection object if the backend returns one.
    """
    if zvec_module is None:
        zvec_module = _get_zvec_module()

    schema = build_collection_schema(name, fields, vectors, zvec_module=zvec_module)

    # Try client first
    if client is not None:
        if hasattr(client, "create_collection"):
            return client.create_collection(schema)
        # some clients expose collections via `create` or `create_collection_if_not_exists`
        for attr in ("create", "create_collection_if_not_exists"):
            if hasattr(client, attr):
                return getattr(client, attr)(schema)

    # Fallback to module-level API
    if hasattr(zvec_module, "create_collection"):
        return zvec_module.create_collection(schema)

    raise RuntimeError(
        "Unable to find a create_collection API on the provided client or the zvec module. "
        "Pass a client with create_collection(schema) or adapt this wrapper to your zvec client."
    )


def list_collections(client: Optional[Any] = None) -> List[str]:
    """Return a list of collection names from the client or zvec module.

    This function tries several common API names.
    """
    zvec_module = None
    try:
        zvec_module = _get_zvec_module()
    except ImportError:
        pass

    if client is not None:
        for attr in ("list_collections", "get_collections", "collections"):
            if hasattr(client, attr):
                res = getattr(client, attr)()
                # normalize result to list[str]
                if isinstance(res, list):
                    return [c if isinstance(c, str) else getattr(c, "name", str(c)) for c in res]
                return [str(res)]

    if zvec_module is not None:
        if hasattr(zvec_module, "list_collections"):
            res = zvec_module.list_collections()
            if isinstance(res, list):
                return [c if isinstance(c, str) else getattr(c, "name", str(c)) for c in res]
    raise RuntimeError("Could not list collections: provide a client or ensure zvec exposes list_collections()")


def get_collection(client: Optional[Any], name: str) -> Any:
    """Fetch a collection object by name from a client or the zvec module.

    Tries several common API names and falls back to raising a helpful error.
    """
    if client is not None:
        for attr in ("get_collection", "collection", "get"):
            if hasattr(client, attr):
                try:
                    return getattr(client, attr)(name)
                except TypeError:
                    # some APIs access client.collection[name]
                    pass
        # try dictionary-like access
        if hasattr(client, "__getitem__"):
            try:
                return client[name]
            except Exception:
                pass

    zvec_module = _get_zvec_module()
    for attr in ("get_collection", "collection", "Collection"):
        if hasattr(zvec_module, attr):
            maybe = getattr(zvec_module, attr)
            try:
                return maybe(name)
            except Exception:
                # try constructing attribute access
                try:
                    return getattr(zvec_module, name)
                except Exception:
                    pass

    raise RuntimeError(
        "Could not resolve collection object. Pass a client with get_collection(name) or adapt this function."
    )


def insert_documents(
    collection_or_client: Any,
    docs: Iterable[Dict[str, Any]],
    *,
    vector_field: str = "local_embedding",
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    batch_size: int = 32,
) -> List[Any]:
    """Insert documents into a collection.

    Args:
        collection_or_client: either a collection object (with insert/insert_many) or a client
            (in which case a collection name string must be provided in each doc via `collection` key).
        docs: iterable of dicts. Each dict must contain at least `id` and `text`. Other keys go to `fields`.
        vector_field: name of the vector field in the collection schema to populate.
        embed_fn: function(text) -> list[float]. Defaults to `local_embed`.

    Returns a list of results returned by the backend insert calls.
    """
    if embed_fn is None:
        embed_fn = local_embed

    zvec_module = None
    try:
        zvec_module = _get_zvec_module()
    except ImportError:
        # we only need zvec when building Doc objects; if collection is already a wrapper
        # that accepts raw dicts, the client may accept them.
        pass

    results = []

    # If the user passed a client and docs include collection name, resolve per-doc.
    # Otherwise assume collection_or_client is a collection object.
    is_collection_obj = hasattr(collection_or_client, "insert") or hasattr(collection_or_client, "insert_many")

    if is_collection_obj:
        collection = collection_or_client
        batch = []
        for d in docs:
            doc_id = d.get("id")
            text = d.get("text")
            if text is None:
                raise ValueError("Each doc dict must contain a 'text' field")
            vec = embed_fn(text)
            fields = {k: v for k, v in d.items() if k not in ("id", "text")}
            if zvec_module is not None:
                doc_obj = zvec_module.Doc(id=doc_id, vectors={vector_field: vec}, fields=fields)
            else:
                # fallback to a simple dict if Doc class isn't available
                doc_obj = {"id": doc_id, "vectors": {vector_field: vec}, "fields": fields}

            # try to batch insert when collection exposes insert_many
            if hasattr(collection, "insert_many"):
                batch.append(doc_obj)
                if len(batch) >= batch_size:
                    results.append(collection.insert_many(batch))
                    batch = []
            else:
                results.append(collection.insert(doc_obj))
        if batch:
            results.append(collection.insert_many(batch)) if hasattr(collection, "insert_many") else None
        return results

    # If we reach here, collection_or_client is assumed to be a client and each doc must name collection
    client = collection_or_client
    for d in docs:
        coll_name = d.get("collection")
        if not coll_name:
            raise ValueError("When passing a client, each doc must include a 'collection' key with name")
        collection = get_collection(client, coll_name)
        # reuse code path for single collection insert
        results.extend(insert_documents(collection, [d], vector_field=vector_field, embed_fn=embed_fn, batch_size=batch_size))
    return results
