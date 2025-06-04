# import os
import streamlit as st
import PyPDF2
import google.generativeai as genai
# from dotenv import load_dotenv
import faiss
import numpy as np

# Load Spark API key from .env
# load_dotenv()
GOOGLE_API_KEY="AIzaSyAk307NWnghCfRpLrzQ_LyESf28AJi_zZ4"
genai.configure(api_key=GOOGLE_API_KEY)

# --- Helper Functions ---

def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_text(texts):
    # Fix: Call genai.embed_content directly, not on a GenerativeModel instance
    # The 'embedding-001' model is designed for embeddings.
    return np.array([
        genai.embed_content(model="models/embedding-001", content=t, task_type="retrieval_document")["embedding"]
        for t in texts
    ]).astype("float32")

def search_similar_chunks(question, chunks, chunk_embeddings, top_k=5):
    index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
    index.add(chunk_embeddings)
    # Fix: Call genai.embed_content directly for query embedding
    q_embed = genai.embed_content(model="models/embedding-001", content=question, task_type="retrieval_query")["embedding"]
    _, indices = index.search(np.array([q_embed]).astype("float32"), top_k)
    return [chunks[i] for i in indices[0]]

def ask_gemini_stream(question, context):
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    prompt = f"""Use the context below to answer the question accurately and concisely:

Context:
{context}

Question: {question}
Answer:"""
    return model.generate_content(prompt, stream=True)

def generate_chat_txt(history):
    text = ""
    for q, a in history:
        text += f"You: {q}\nSpark: {a}\n\n"
    return text.encode("utf-8")

# --- Streamlit App ---

st.set_page_config("📄 Spark AI PDF Chatbot", layout="wide")
st.title("📄 Spark 2.5 Rapid PDF Chatbot By Aaradhya Vanakhade")
st.caption("Ask questions across multiple PDFs. Powered by Spark 2.5 Rapid.")

# Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

# PDF Upload
pdf_files = st.file_uploader("📄 Upload PDF(s)", type="pdf", accept_multiple_files=True)

if pdf_files and st.button("📚 Process PDFs"):
    with st.spinner("🔍 Extracting and embedding PDFs..."):
        full_text = extract_text_from_pdfs(pdf_files)
        st.session_state.pdf_text = full_text
        st.session_state.chunks = chunk_text(full_text)
        st.session_state.embeddings = embed_text(st.session_state.chunks)
    st.success("✅ PDFs processed and ready for questions!")

# Control Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑 Clear Chat"):
        st.session_state.history = []
with col2:
    if st.session_state.history:
        download_txt = generate_chat_txt(st.session_state.history)
        st.download_button("📥 Download Chat History", data=download_txt, file_name="chat_history.txt")

# Question Input & Response
if st.session_state.embeddings is not None:
    user_input = st.text_input("💬 Ask a question about the PDFs:")
    if user_input:
        matched_chunks = search_similar_chunks(user_input, st.session_state.chunks, st.session_state.embeddings)
        context = "\n\n".join(matched_chunks)

        st.markdown("**🤖 Spark is typing...**")
        response_placeholder = st.empty()
        full_response = ""

        for chunk in ask_gemini_stream(user_input, context):
            if chunk.candidates:
                part = chunk.candidates[0].content.parts[0].text
                full_response += part
                response_placeholder.markdown(full_response)

        st.session_state.history.append((user_input, full_response))

# Display Chat History
if st.session_state.history:
    st.divider()
    st.subheader("💬 Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**🟢 You:** {q}")
        st.markdown(f"**🤖 Spark:** {a}")
        st.markdown("---")