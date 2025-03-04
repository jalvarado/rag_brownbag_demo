import os
import shutil
import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from get_embedding_function import get_embedding_function


DATA_PATH = "./data"
CHROMA_PATH = "./chroma"
PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}
Question: {question}
"""


def get_chroma_db():
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


def reset_vector_index():
    db = get_chroma_db()
    db.delete_collection()


def load_documents():
    document_loader = DirectoryLoader(DATA_PATH, show_progress=True)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def load_vector_index():
    with st.spinner("Loading documents to vector store..."):
        db = get_chroma_db()
        documents = load_documents()
        chunks = split_documents(documents)
        chunks_with_ids = calculate_chunk_ids(chunks)
        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")


def search_vector_store(query_text, k=5):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    results = db.similarity_search_with_score(query_text, k)
    return results


def query_without_rag(query_text, model_name):
    model = Ollama(model=model_name, temperature=0.0, top_k=5, top_p=0.5)
    return model.invoke(query_text)


def query_with_rag(query_text, model_name):
    model = Ollama(model=model_name, temperature=0.0, top_k=5, top_p=0.5)
    similarity_results = search_vector_store(query_text, 5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in similarity_results])

    # format will replace the '{keyword}' fields in the prompt template
    # with the values passed to the function.
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    return model.invoke(prompt)


def reset_conversation_history():
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


# ===== Streamlit code ===== #
st.set_page_config(page_title="RAG Brown Bag", layout="wide")
st.title("RAG Brown Bag")

use_rag = st.sidebar.toggle("Use RAG")
model_name = st.sidebar.selectbox("Model Name", ["mistral", "llama3"])
st.sidebar.button("Reset Conversation", on_click=reset_conversation_history)

if os.path.exists(CHROMA_PATH):
    st.sidebar.button("Clear RAG Data", on_click=reset_vector_index)

st.sidebar.button("Load Data", on_click=load_vector_index)


if "messages" not in st.session_state:
    reset_conversation_history()

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        if use_rag:
            print("Using RAG to add context to the prompt")
            response = query_with_rag(prompt, model_name=model_name)
        else:
            print("Querying the LLM without RAG")
            response = query_without_rag(prompt, model_name=model_name)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
