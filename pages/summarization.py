import streamlit as st
import fitz  # PyMuPDF
import docx2txt
from transformers import pipeline

# Function to read PDF and extract text
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to read DOCX and extract text
def read_docx(file):
    return docx2txt.process(file)

# Function to summarize text using T5 model
def summarize_text_t5(text, summarizer):
    try:
        if not text.strip():
            return ""
        
        # Split the text into chunks that fit the model's max_length
        max_chunk_size = 512
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

def app():
    st.title("Text Summarization")

    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "pdf":
            with st.spinner("Reading PDF..."):
                document_text = read_pdf(uploaded_file)
        elif file_type == "docx":
            with st.spinner("Reading DOCX..."):
                document_text = read_docx(uploaded_file)

        summarizer_t5 = pipeline("summarization", model="t5-small")
        with st.spinner("Summarizing text..."):
            summary_t5 = summarize_text_t5(document_text, summarizer_t5)
        if summary_t5:
            st.subheader("Summary:")
            st.write(summary_t5)
