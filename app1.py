import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector embedding
def create_vector_embedding():
    if "vectors" not in st.session_state:
        # Initialize embeddings and vector database
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # Data ingestion step
        st.session_state.docs = st.session_state.loader.load()  # Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit UI
st.title("RAG Document Q&A With Groq and Llama3")

# User input
user_prompt = st.text_input("Enter your query from the research paper")

# Button to initialize vector embedding
if st.button("Document Embedding"):
    create_vector_embedding()
    st.success("Vector Database is ready!")

# Handle user query
if user_prompt:
    # Check if vectors are initialized
    if "vectors" not in st.session_state:
        st.error("Vector database is not initialized. Please click 'Document Embedding' first.")
    else:
        # Create retrieval and document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"Response time: {time.process_time() - start:.2f} seconds")

        # Display answer
        st.write(response['answer'])

        # Display similar documents
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
