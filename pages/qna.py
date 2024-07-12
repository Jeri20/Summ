import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset

# Load the tokenizer and retriever
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Load the dataset with trust_remote_code=True
dataset = load_dataset("wiki_dpr", split="train", trust_remote_code=True)

# Load the retriever with the dataset
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True, dataset=dataset)

# Load the model
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

st.title("Simple Q&A with RAG")
st.write("Ask a question and get an answer!")

question = st.text_input("Enter your question:")
if question:
    inputs = tokenizer(question, return_tensors="pt")
    # Retrieve documents
    doc_scores, doc_titles = retriever.retrieve(question, n_docs=5)
    inputs['context_input_ids'] = tokenizer(doc_titles, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids']
    inputs['context_attention_mask'] = tokenizer(doc_titles, return_tensors="pt", padding=True, truncation=True, max_length=512)['attention_mask']
    
    # Generate the answer
    outputs = model.generate(**inputs, num_beams=4, num_return_sequences=1)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    st.write(f"Answer: {answer}")
