import streamlit as st
from transformers import pipeline

st.title("Simple Q&A with Transformers")
st.write("Ask a question and get an answer!")

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the question-answering pipeline
    qa_pipeline = pipeline("question-answering")
    return qa_pipeline

qa_pipeline = load_model()

question = st.text_input("Enter your question:")
context = st.text_area("Enter the context (optional):", height=200)

if st.button("Get Answer"):
    if question:
        if context:
            result = qa_pipeline(question=question, context=context)
        else:
            # Use a default context if none is provided
            default_context = "The capital of France is Paris. The capital of Germany is Berlin."
            result = qa_pipeline(question=question, context=default_context)
        st.write(f"Answer: {result['answer']}")
    else:
        st.write("Please enter a question.")
