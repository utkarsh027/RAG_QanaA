# RAG-Based Q&A App with Streamlit

Here’s a detailed README.md for your Q&A app using RAG, OpenAI embeddings, PDF retrieval, and Streamlit:

RAG-Based Q&A App with Streamlit
This project implements a Retrieval-Augmented Generation (RAG) approach for creating a Q&A chatbot. The app uses OpenAI embeddings to process and retrieve information from a PDF document (attention.pdf) and leverages Streamlit to provide an interactive user interface.

The system enables users to ask questions about the content of the PDF and receive accurate responses based on the document.

# FEATURES:

PDF Document Retrieval: Uses attention.pdf as the primary knowledge source.
OpenAI Embeddings: Embeds PDF content for effective retrieval of relevant context.
RAG Architecture: Combines document retrieval and language generation for precise answers.
Interactive Interface: Built with Streamlit for a smooth user experience.
Customizable API Keys: Reads OpenAI API keys securely using environment variables.
# Prerequisites
Python 3.10 or higher
OpenAI API Key (Get one from OpenAI)
Required Python libraries (see requirements.txt)

# Follow this

# Step 1: Clone the Repository
# Step 2: Create a Virtual Environment
# Step 3: Install Dependencies
pip install -r requirements.txt
# Step 4: Configure API Keys
echo "OPENAI_API_KEY='your_openai_api_key_here'" > .env

# Running the App
python -m streamlit run app1.py
