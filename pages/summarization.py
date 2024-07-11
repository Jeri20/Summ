import streamlit as st
from transformers import pipeline

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

# Main function for the summarization page
def summarization_page():
    st.title("Text Summarization with T5")

    # Upload document
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        # Read PDF or DOCX and extract text
        if file_type == "pdf":
            with st.spinner("Reading PDF..."):
                document_text = read_pdf(uploaded_file)
        elif file_type == "docx":
            with st.spinner("Reading DOCX..."):
                document_text = read_docx(uploaded_file)
        
        # Initialize T5 summarizer pipeline
        summarizer_t5 = pipeline("summarization", model="t5-small")

        # Summarize the extracted text
        with st.spinner("Summarizing text..."):
            summary_t5 = summarize_text_t5(document_text, summarizer_t5)
        
        if summary_t5:
            st.subheader("Summary (T5):")
            st.write(summary_t5)

if __name__ == "__main__":
    summarization_page()
