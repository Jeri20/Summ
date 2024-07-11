import streamlit as st
import fitz  # PyMuPDF
import docx2txt
from transformers import pipeline, RagTokenizer, RagRetriever, RagSequenceForGeneration

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

# Function to summarize text using a Language Model
def summarize_text(text, summarizer):
    try:
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

# Streamlit app
def main():
    st.title("Text Summarization and Q&A with RAG")

    # Upload document
    uploaded_file = st.file_uploader("Upload a PDF or DOCX file", type=["pdf", "docx"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type == "pdf":
            with st.spinner("Reading PDF..."):
                document_text = read_pdf(uploaded_file)
        elif file_type == "docx":
            with st.spinner("Reading DOCX..."):
                document_text = read_docx(uploaded_file)

        # Summarizer pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Summarize the extracted text
        with st.spinner("Summarizing text..."):
            summary = summarize_text(document_text, summarizer)
        if summary:
            st.subheader("Summary:")
            st.write(summary)

            # Initialize RAG model
            tokenizer, model = initialize_rag()

            # Ask question
            question = st.text_input("Ask a question about the document")

            if question:
                with st.spinner("Getting answer..."):
                    answer = get_answer(question, document_text, tokenizer, model)
                if answer is not None:
                    st.subheader("Answer:")
                    st.write(answer)

if __name__ == "__main__":
    main()
