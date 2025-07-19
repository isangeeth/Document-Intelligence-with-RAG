import os
import sys
import tempfile
import fitz  # PyMuPDF
import streamlit as st
import faiss
import numpy as np
import subprocess
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# ------------------------
# Utility to list Ollama models
# ------------------------
def get_ollama_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")[1:]  # skip header
        models = [line.split()[0] for line in lines if line]
        return models
    except FileNotFoundError:
        return None

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Document Q&A - RAG App")
st.title("📄 Document Q&A using RAG")

# Select LLM Source
llm_option = st.selectbox("Choose LLM Backend", ["OpenAI", "Ollama (Local)"])
api_key = None
selected_ollama_model = None
ollama_models = []

if llm_option == "OpenAI":
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
else:
    models = get_ollama_models()
    if models is None:
        st.warning("⚠️ Ollama is not installed or not accessible. Please install it from https://ollama.com")
    elif not models:
        st.warning("⚠️ No Ollama models found. Run `ollama pull mistral` or another model to install one.")
    else:
        selected_ollama_model = st.selectbox("Select Ollama Model", models)

# Upload PDF Files using file browser
uploaded_files = st.file_uploader("Upload PDF document(s)", type=["pdf"], accept_multiple_files=True, label_visibility="visible")

# Ask question
question = st.text_input("Ask a question about the document (Type 'goodbye' to exit)")

# Exit if 'goodbye' is typed
if question.lower().strip() == "goodbye":
    st.success("👋 Goodbye! Shutting down the app...")
    st.stop()
    os._exit(0)  # force stop the script

# Run button
if st.button("Get Answer") and uploaded_files and question:
    # ------------------------
    # Extract Text from PDFs
    # ------------------------
    full_text = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        for page in doc:
            full_text += page.get_text()
        doc.close()
        os.remove(tmp_path)

    # ------------------------
    # Chunk Text
    # ------------------------
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(full_text)
    documents = [Document(page_content=txt) for txt in texts]

    # ------------------------
    # Generate Embeddings
    # ------------------------
    if llm_option == "OpenAI":
        if not api_key:
            st.warning("Please enter your OpenAI API key.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = api_key
        embeddings = OpenAIEmbeddings()
    else:
        if not selected_ollama_model:
            st.warning("No model selected from Ollama.")
            st.stop()
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector DB
    vectorstore = FAISS.from_documents(documents, embeddings)

    # ------------------------
    # LLM Setup
    # ------------------------
    if llm_option == "OpenAI":
        llm = OpenAI(temperature=0.2)
    else:
        llm = Ollama(model=selected_ollama_model)

    # RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # ------------------------
    # Run QA
    # ------------------------
    with st.spinner("Generating answer..."):
        result = qa_chain({"query": question})
        st.subheader("🧠 Answer")
        st.write(result['result'])

        # Show source chunks
        with st.expander("📚 Source Chunks"):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)