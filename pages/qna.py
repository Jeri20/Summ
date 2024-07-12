import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset

st.title("Simple Q&A with RAG")
st.write("Ask a question and get an answer!")

@st.cache_resource(allow_output_mutation=True)
def load_rag_model():
    # Load the tokenizer and retriever
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
    
    # Load the dataset with trust_remote_code=True
    dataset = load_dataset("wiki_dpr", split="train", trust_remote_code=True)
    
    # Load the retriever with the dataset
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-sequence-nq", 
        index_name="exact", 
        passages_path=None, 
        dataset=dataset
    )
    
    # Load the model
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
    
    return tokenizer, retriever, model

tokenizer, retriever, model = load_rag_model()

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        inputs = tokenizer(question, return_tensors="pt")
        
        # Retrieve documents
        question_inputs = tokenizer(question, return_tensors="pt")
        retrieved_docs = retriever(question_inputs.input_ids)
        context_input_ids = retrieved_docs['context_input_ids']
        context_attention_mask = retrieved_docs['context_attention_mask']
        
        inputs['context_input_ids'] = context_input_ids
        inputs['context_attention_mask'] = context_attention_mask

        # Generate the answer
        outputs = model.generate(**inputs, num_beams=4, num_return_sequences=1)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question.")
