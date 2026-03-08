import streamlit as st
from transformers import pipeline
import urllib.parse

# 1. Page Configuration (Must be at the very top)
st.set_page_config(page_title="AI Question and Answering app", layout="wide")


# 2. Optimized Model Loading with Caching
@st.cache_resource
def load_model():
    # Specifically using the mdeberta model for your QA task
    return pipeline(
        task="question-answering",
        model="timpal0l/mdeberta-v3-base-squad2"
    )


qa_model = load_model()

# 3. Main Interface
st.title("AI Powered Question and Answering webApp")

# 4. Columns Setup
col0, col1, col2 = st.columns([2, 1, 1])

# 5. INPUTS (Defined in col0 so variables are available for subsequent blocks)
with col0:
    st.subheader("Input Data")
    context = st.text_area("Text area for context",
                           height=150,
                           placeholder="Enter Context here....")

    question = st.text_area("Text area for question",
                            height=100,
                            placeholder="Ask your question here")

    submit_btn = st.button("Ask Question", type="primary")

# 6. SEARCH LOGIC (Uses the 'question' variable defined above)
with col1:
    if question.strip():
        # URL encode the question for a Google search link
        query = urllib.parse.quote(question)
        google_url = f"https://www.google.com/search?q={query}"

        st.markdown("### 🔎 Search Context")
        st.info("Verify your query on Google:")
        st.markdown(f"[Search on Google for: **{question}**]({google_url})")
    else:
        st.info("Waiting for a question...")

# 7. BRANDING
with col2:
    st.markdown("### Credits")
    st.markdown("Powered by C-Clarke Institute students")

# 8. EXECUTION LOGIC (Runs only when the button is clicked)
if submit_btn:
    if context.strip() and question.strip():
        with st.spinner("Answering your questions........"):
            try:
                # Perform Question Answering
                result = qa_model(question=question, context=context)

                st.divider()
                st.success("### Answer Found:")
                st.write(result['answer'])

                # Show Confidence Score
                st.metric("Confidence Score", f"{round(result['score'] * 100, 2)}%")
            except Exception as e:
                st.error(f"Processing error: {e}")
    else:
        st.warning("Please enter both context and a question.")