import os 
import streamlit as st 
from langchain_groq import ChatGroq 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate 
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS #### vectorstore db 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embeding technique 

from dotenv import load_dotenv 

load_dotenv() 

## load the groq and the Google Api variable  
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") 

st.title("Gemma Model Document Chat")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it") 

