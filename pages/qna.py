import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset

# Load the tokenizer and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)

# Load the model
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Load the dataset with trust_remote_code=True
dataset = load_dataset("wiki_dpr", split="train", trust_remote_code=True)

st.title("Simple Q&A with RAG")
st.write("Ask a question and get an answer!")

question = st.text_input("Enter your question:")
if question:
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=4, num_return_sequences=1)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    st.write(f"Answer: {answer}")
