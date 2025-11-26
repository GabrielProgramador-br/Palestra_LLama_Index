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
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
import chromadb
import pymupdf


# -------------------------------------
# CONFIGURAÇÃO STREAMLIT
# -------------------------------------
st.set_page_config(page_title="IA com LlamaIndex + Llama 3", layout="wide")
st.title("🤖 IA com LlamaIndex + Llama 3 (Groq) + PDFs")


# -------------------------------------
# CARREGAR A GROQ_API_KEY DO SECRETS
# -------------------------------------
if "GROQ_API_KEY" not in st.secrets:
    st.error("🚨 Adicione GROQ_API_KEY em Settings → Secrets no Streamlit Cloud.")
    st.stop()

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


# -------------------------------------
# CONFIGURAÇÃO DO LLM (Groq + Llama 3)
# -------------------------------------
llm = Groq(model="llama3-70b-8192")

# Forçar o uso do Groq e desativar OpenAI
Settings.llm = llm
Settings.ai_model = None


# -------------------------------------
# CONFIGURAÇÃO DO EMBEDDER HF
# -------------------------------------
embed_model = HuggingFaceEmbedding("sentence-transformers/all-mpnet-base-v2")
Settings.embed_model = embed_model


# -------------------------------------
# UPLOAD DOS PDFs
# -------------------------------------
uploaded_files = st.file_uploader(
    "📄 Faça upload de PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    os.makedirs("pdfs", exist_ok=True)

    for file in uploaded_files:
        with open(f"pdfs/{file.name}", "wb") as f:
            f.write(file.read())

    st.success("📁 PDFs carregados com sucesso!")


    # -------------------------------------
    # CHROMA DB (base vetorial persistente)
    # -------------------------------------
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection("llama3_index")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vector_store)


    # -------------------------------------
    # LEITURA DOS PDFs
    # -------------------------------------
    docs = SimpleDirectoryReader("pdfs").load_data()


    # -------------------------------------
    # CRIAÇÃO DO ÍNDICE
    # -------------------------------------
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage
    )


    # -------------------------------------
    # CRIAÇÃO DO QUERY ENGINE (modo seguro)
    # -------------------------------------
    retriever = index.as_retriever(similarity_top_k=5)
    response_synthesizer = get_response_synthesizer(llm=llm)

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )


    # -------------------------------------
    # PERGUNTA DO USUÁRIO
    # -------------------------------------
    question = st.text_input("❓ Faça uma pergunta sobre seus PDFs:")

    if question:
        with st.spinner("Consultando o modelo..."):
            answer = query_engine.query(question)

        st.subheader("📌 Resposta")
        st.write(answer)
