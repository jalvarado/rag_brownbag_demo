import chromadb


client = chromadb.PersistentClient(path="chroma/")

# List all collections
print("Listing all collections: ")
collections = client.list_collections()
[print(c) for c in collections]

# Inspect the langchain collection
langchain_collection = client.get_collection(name="langchain")
print(f"langchain collection has {langchain_collection.count()} embeddings.")

# Get the fist 10 embeddings
batch = langchain_collection.get(
    include=["metadatas", "documents", "embeddings"], limit=10, offset=0
)
print("Metadata:")
print(batch["metadatas"][0])

print("Embedding:")
print(batch["embeddings"][0])
