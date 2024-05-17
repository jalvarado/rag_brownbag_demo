import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from get_embedding_function import get_embedding_function


CHROMA_PATH = "./chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def search_vector_store(query_text, k=5):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

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
st.sidebar.button("Clear conversation history", on_click=reset_conversation_history)

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
