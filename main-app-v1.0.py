## functional dependencies
import time
import streamlit as st

## setting up env
import os
from dotenv import load_dotenv
from numpy.core.defchararray import endswith
load_dotenv()

## LangChain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate

## LCEL implementation of LangChain ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ## setting up file paths
# current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(current_dir, "data")
# persistent_directory = os.path.join(current_dir, "data-ingestion-local")

# ## initializing the UI
# st.set_page_config(page_title="RAG-Based Legal Assistant")
# col1, col2, col3 = st.columns([1, 25, 1])
# with col2:
#     st.title("RAG-Based Legal Assistant")

# ## setting-up the LLM
# chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)

# ## setting up -> streamlit session state
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# if "memory" not in st.session_state:
#     st.session_state['memory'] = ConversationSummaryMemory(llm=ChatGroq(model="gemma-7b-it", temperature=0.2))

# ## resetting the entire conversation
# def reset_conversation():
#     st.session_state['messages'] = []
#     st.session_state['memory'].clear()

# ## open-source embedding model from HuggingFace - taking the default model only
# embedF = HuggingFaceEmbeddings()

# ## loading the vector database from local
# vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)

# ## setting up the retriever
# kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 3})

# condense_question_system_template = (
#     "Based on the provided summary of the chat history and the most recent user question, "
#     "which may reference elements from the chat history, please reformulate the question into "
#     "a clear and standalone format that can be comprehended independently of the chat history. "
#     "Do not provide an answer to the question; "
#     "simply return the reformulated question or the original "
#     "question if no changes are necessary."
# )

# condense_question_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", condense_question_system_template),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#     ]
# )

# history_aware_retriever = create_history_aware_retriever(
#     llm = chatmodel,
#     retriever = kb_retriever,
#     prompt = condense_question_prompt
# )

# ## setting-up the prompt
# main_system_prompt_template = (
#     "<s>[INST] This is a chat template. As a Legal Assistant Chatbot specializing in legal queries, "
#     "your primary objective is to provide accurate and concise information based on user queries. "
#     "You will adhere strictly to the instructions provided, offering relevant "
#     "context from the knowledge base while avoiding unnecessary details. "
#     "Your responses will be brief, to the point, concise and in compliance with the established format. "
#     "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
#     "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
#     "\nCONTEXT: {context}\nCHAT HISTORY: {chat_history} \nQUESTION: {question}"
#     "\nANSWER:\n</s>[INST]"
# )

# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", main_system_prompt_template),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#     ]
# )

# qa_chain = create_stuff_documents_chain(
#     llm=chatmodel,
#     prompt=qa_prompt
# )

# convo_qa_chain = create_retrieval_chain(
#     retriever=history_aware_retriever,
#     combine_docs_chain = qa_chain
# )

# ## setting-up the chain
# st.write(prompt)