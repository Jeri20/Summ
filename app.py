import streamlit as st
import fitz  # PyMuPDF
import docx2txt
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, pipeline

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
        # Ensure text is not empty or too short
        if not text.strip():
            return ""
        
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

# Function to initialize RAG components
def initialize_rag():
    try:
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", trust_remote_code=True)
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever, trust_remote_code=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        st.stop()

# Function to get answer from RAG model
def get_answer(question, context, tokenizer, model):
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        generated = model.generate(**inputs)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return None

# Main Streamlit app logic
def main():
    st.title("Text Summarization and Q&A with RAG")

    # Page 1: File Upload and Text Extraction
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "pdf":
            with st.spinner("Reading PDF..."):
                document_text = read_pdf(uploaded_file)
        elif file_type == "docx":
            with st.spinner("Reading DOCX..."):
                document_text = read_docx(uploaded_file)

        # Page 2: Text Summarization using T5
        summarizer_t5 = pipeline("summarization", model="t5-small")
        with st.spinner("Summarizing text..."):
            summary_t5 = summarize_text_t5(document_text, summarizer_t5)
        if summary_t5:
            st.subheader("Summary (T5):")
            st.write(summary_t5)

            # Page 3: Question Answering with RAG
            tokenizer, model = initialize_rag()
            question = st.text_input("Ask a question about the document")
            if question:
                with st.spinner("Getting answer..."):
                    answer = get_answer(question, document_text, tokenizer, model)
                if answer is not None:
                    st.subheader("Answer:")
                    st.write(answer)

if __name__ == "__main__":
    main()
