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
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context:
    <context>
    {context} 
    <context>
    Question: {input}
    """)

def vector_embeding():

    if "vectors" not in st.session_state:
        st.session_state.embedding = GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./Pdfs")   #data injesttion
        st.session_state.docs = st.session_state.loader.load() #document Loading 
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embedding)


prompt1 = st.text_input("Enter your Question")

if st.button("creating Vectore Store"):
    vector_embeding()
    st.write("Vector Store DB is Ready !! Created")


import time 

if prompt1: 
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever =st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})

    st.write(response['answer'])

    ### with a stream;it expander  
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")