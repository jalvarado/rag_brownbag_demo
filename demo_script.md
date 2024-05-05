# RAG Demo

## Preparing Data

### Splitting Large Files

There is an optimal size for embeddings that allow for accurate and efficient nearest
neighbor search retrieval. This optimal size is most likely going to be much smaller
than most documents being indexed. Because of this we have to split the documents up
into smaller chunks before running those chunks through the embedding model. However,
when splitting large documents into chunks, you want to try your best to keep semantically
similar text together.

Langchain implements a variety of text splitters that use different splitting
strategeis aimed at keeping semantically similar data together.

[Langchain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

### Calculate Chunk ID

To avoid regenerating embeddings that have already been calculated, we generate
a chunk ID based on the document name, page number, and chunk index for the
document

### Generating Embeddings
