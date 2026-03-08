
import  streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Question and Answering app")

@st.cache_resource
def load_model():
    return pipeline(task="question-answering", model="timpal0l/mdeberta-v3-base-squad2")

qa_model = load_model()

st.title("AI Powered Question and Answering webApp")

col1, col2 = st.columns([2,1])

with col1:
    context = st.text_area("Text area for context",
                           height=150, placeholder="Enter Context here....")
    question = st.text_area("Text area for question", height=150,placeholder="Ask your question here")
    submit_btn = st.button("Ask Question", type="primary")

with col2:
    st.markdown("Powered by C-Clarke Institute students")

if context and question and submit_btn:
    with st.spinner("Answering your questions........"):
        result = qa_model(question=question,context=context)
        st.metric(result['score'])
        st.metric("Confidence Score", round(result["score"], 3))

else:
    st.markdown("invalid Input....")
