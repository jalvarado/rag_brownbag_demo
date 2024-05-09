# RAG Demo

## Preparing Data
Before we can query the vector store for relevant documents, we must first prepare
the data to be converted to vectors with semantic meaning.  To do this we can run
the _prepare_data.py_ script that will check for data files in the _data/_ directory
and generate embeddings to be saved into a local ChromaDB instance.

```sh
python prepare_data.py`
```

To clean the database and remove old embeddings before generating the embeddings run:

```sh
python prepare_data.py --reset
```

### Splitting Large Files

There is an optimal size for embeddings that allow for accurate and efficient nearest
neighbor search retrieval without losing semantic meaning. This optimal size is most 
likely going to be much smaller than most documents being indexed. Because of this we 
have to split the documents up into smaller chunks before running those chunks through 
the embedding model. However, when splitting large documents into chunks, you want to 
try your best to keep semantically similar text together.

Langchain implements a variety of text splitters that use different splitting
strategeis aimed at keeping semantically similar data together.

[Langchain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

### Calculate Chunk ID

To avoid regenerating embeddings that have already been calculated, we generate
a chunk ID based on the document name, page number, and chunk index for the
document

### Generating Embeddings
