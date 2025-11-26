import streamlit as st
import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import pymupdf


st.set_page_config(page_title="IA com LlamaIndex + Llama3", layout="wide")
st.title("🤖 IA com LlamaIndex + Llama 3 (Groq) + PDFs")


os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


llm = Groq(model="llama3-70b-8192")

embed_model = HuggingFaceEmbedding("sentence-transformers/all-mpnet-base-v2")
Settings.embed_model = embed_model
Settings.llm = llm


uploaded_files = st.file_uploader("📄 Faça upload de PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("pdfs", exist_ok=True)

    for file in uploaded_files:
        with open(f"pdfs/{file.name}", "wb") as f:
            f.write(file.read())

    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("llama3_index")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)

    docs = SimpleDirectoryReader("pdfs").load_data()

    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage
    )

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        llm=llm
    )

    question = st.text_input("❓ Faça uma pergunta sobre seus PDFs:")

    if question:
        answer = query_engine.query(question)
        st.subheader("📌 Resposta")
        st.write(answer)
