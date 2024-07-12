import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Function to initialize RAG components
def initialize_rag():
    try:
        # Initialize tokenizer
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        
        # Initialize retriever with trust_remote_code=True
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-sequence-nq",
            index_name="exact",
            use_dummy_dataset=True,
            trust_remote_code=True
        )
        
        # Initialize model with retriever and trust_remote_code=True
        model = RagSequenceForGeneration.from_pretrained(
            "facebook/rag-sequence-nq",
            retriever=retriever,
            trust_remote_code=True
        )
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        return None, None

# Function to get answer from RAG model
def get_answer(question, context, tokenizer, model):
    try:
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
        generated = model.generate(**inputs)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return None

# Main function for the QA page
def qa_page():
    st.title("Question Answering with RAG")

    # Initialize RAG components
    tokenizer, model = initialize_rag()

    if tokenizer is not None and model is not None:
        # User inputs question and context
        question = st.text_input("Ask a question")
        context = st.text_area("Context (document or article)")

        if question and context:
            # Get answer from RAG model
            with st.spinner("Getting answer..."):
                answer = get_answer(question, context, tokenizer, model)

            if answer is not None:
                st.subheader("Answer:")
                st.write(answer)

if __name__ == "__main__":
    qa_page()
