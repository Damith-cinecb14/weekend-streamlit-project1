import  streamlit as st
from streamlit import subheader
from transformers import pipeline
import urllib.parse

st.set_page_config(page_title="AI Question and Answering app", layout="wide")

@st.cache_resource
def load_model():
    return pipeline(
        task="question-answering",
        model="timpal0l/mdeberta-v3-base-squad2"
    )


qa_model = load_model()

st.title("AI Powered Question and Answering webApp")

col0, col1, col2 = st.columns([2,1,1])



with col0:
    subheader("Input Data")
    context = st.text_area("Text area for context",
                           height=150, placeholder="Enter Context here....")
    question = st.text_area("Text area for question", height=150,placeholder="Ask your question here")
    submit_btn = st.button("Ask Question", type="primary")

with col1:
    if question:
        query = urllib.parse.quote(question)
        google_url = f"https://www.google.com/search?q={query}"

        st.markdown("### 🔎 Search Context")
        st.info("You can verify facts or find more info on Google:")
        st.markdown(f"[Search on Google]({google_url})")
    else:
        st.info("Waiting for a question to generate search links...")


with col2:
    st.markdown("### Credits")
    st.markdown("Powered by C-Clarke Institute students")

if submit_btn:
    if context.strip() and question.strip():
        with st.spinner("Answering your questions........"):
            try:
                # Perform Question Answering
                result = qa_model(question=question, context=context)

                st.divider()
                st.success("### Answer Found:")
                st.write(result['answer'])

                # Show confidence score
                st.metric("Confidence Score", f"{round(result['score'] * 100, 2)}%")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
    else:
        st.warning("Please provide both a **Context** and a **Question** before clicking the button.")

