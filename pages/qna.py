import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Function to initialize RAG components with trust_remote_code=True
def initialize_rag():
    try:
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", trust_remote_code=True)
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", trust_remote_code=True)
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever, trust_remote_code=True)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        st.stop()

# Function to get an answer from the RAG model
def get_answer(question, context, tokenizer, model):
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        generated = model.generate(**inputs)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return None

def app():
    st.title("Q&A with RAG")

    context = st.text_area("Enter the document text or summary here:")
    
    if context:
        tokenizer, model = initialize_rag()
        question = st.text_input("Ask a question about the document")
        if question:
            with st.spinner("Getting answer..."):
                answer = get_answer(question, context, tokenizer, model)
            if answer is not None:
                st.subheader("Answer:")
                st.write(answer)
