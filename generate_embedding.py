import ollama

EMBEDDING_MODEL = "nomic-embed-text"

response = ollama.embeddings(
    prompt="This is an example sentence used for embedding.", model=EMBEDDING_MODEL
)
print(f"Embedding vector dimensions: {len(response['embedding'])}")
print("--------")
# print(response["embedding"])
