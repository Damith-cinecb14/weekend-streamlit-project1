AI-Powered Extractive QA WebApp
This repository contains a specialized Natural Language Processing (NLP) application designed to find exact answers within a given context. Unlike standard chatbots, this tool uses a dedicated mDeBERTa-v3 engine to pinpoint factual information with high accuracy.

Key Features
Context-Based Retrieval: Users provide a reference text (context), and the AI extracts the specific answer to a related question.

Performance Optimization: Utilizes @st.cache_resource to ensure the heavy AI model is loaded into memory only once, making the app much faster for repeated use.

Confidence Metrics: Every answer includes a "Confidence Score," helping users understand how likely the extracted text is to be correct.

Clean Interface: A responsive two-column layout built with Streamlit for a professional user experience.

Component,Technology

Frontend,Streamlit
Model Engine,timpal0l/mdeberta-v3-base-squad2
Framework,Hugging Face Transformers
Caching,Streamlit Resource Caching

Getting Started

Clone the Repository:

git clone https://github.com/your-username/your-repo-name.git
Install Dependencies:
(Note: Ensure your filename is requirements.txt so pip can find it.)

pip install -r requirements.txt
Run the App:

streamlit run app.py