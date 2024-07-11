import streamlit as st
from multiapp import MultiApp
from pages import summarization, qna

app = MultiApp()

# Add all your application here
app.add_app("Summarization", summarization.app)
app.add_app("Q&A with RAG", qna.app)

# The main app
app.run()
