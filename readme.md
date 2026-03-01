```markdown
pip install -U sentence-transformers

```

## ZvecSample - zvec_service helper

This project contains example scripts for working with Zvec and a small service
wrapper `src/zvec_service.py` which provides:

- lazy local embeddings via `sentence-transformers`
- helpers to build `CollectionSchema` objects from simple descriptors
- wrapper helpers to create/list/get collections and to insert documents
	(these helpers try several common zvec client APIs; pass your client when available)

Quick start

1. Install dependencies (recommended in a virtualenv):

```bash
python -m pip install -r requirements.txt
```

2. Run the demo (it will run local embedding and build a schema object):

```bash
python src\demo_zvec_service.py
```

3. Use `src/zvec_service.py` in your app. Example (conceptual):

```python
from zvec import Client  # your actual zvec client
from zvec_service import create_collection, insert_documents

client = Client(...)  # create/connect your client
create_collection(client, name="documents", fields=[{"name":"title"},{"name":"text"}], vectors=[{"name":"local_embedding","dimension":384}])

# Insert docs
insert_documents(client, [{"collection":"documents","id":"1","text":"Hello world","title":"Hi"}])
```

Notes
- The wrappers are defensive and may need small adjustments to match the exact
	zvec client API you run. If you prefer, pass a collection object (not a client)
	into `insert_documents` and the helper will call `insert`/`insert_many` on it.

- See the official Zvec docs for server and client setup: https://zvec.org/en/docs/

