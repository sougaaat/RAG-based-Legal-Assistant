## functional libraries
import time
import streamlit as st

## setting up env
import os
from dotenv import load_dotenv
from numpy.core.defchararray import endswith
load_dotenv()

## langchain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

## setting up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

## setting up the UI
st.set_page_config(page_title="RAG-Based Legal Assistant")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Legal Assistant")

## setting-up the LLM
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)

## setting up -> streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationSummaryMemory(llm=chatmodel)

## resetting the entire conversation
def reset_conversation():
    st.session_state['messages'] = []
    st.session_state['memory'].clear()

## open-source embedding model from HuggingFace - taking the default model only
embedF = HuggingFaceEmbeddings()

## loading the vector database from local
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)

## setting up the retriever
kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 5})

## setting-up the prompt
promptTemplate = """<s>[INST] This is a chat template. As a Legal Assistant Chatbot specializing in legal queries,
your primary objective is to provide accurate and concise information based on user queries. 
You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. 
Your responses will be brief, to the point, concise and in compliance with the established format. 
If a question falls outside the given context, you will simply output that you are sorry and you don't know about this.
The aim is to deliver professional, precise, and contextually relevant information pertaining to the context.
CONTEXT: {context}
CHAT HISTORY SUMMARY: {chat_history_summary}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=promptTemplate, input_variables=['context', 'chat_history_summary', 'question'])

## setting-up the chain
st.write(prompt)