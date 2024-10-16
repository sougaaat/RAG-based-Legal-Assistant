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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere.chat_models import ChatCohere
## LCEL implementation of LangChain ConversationalRetrievalChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

## setting up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

## initializing the UI
st.set_page_config(page_title="RAG-Based Legal Assistant")
col1, col2, col3 = st.columns([1, 25, 1])
with col2:
    st.title("RAG-Based Legal Assistant")

## setting-up the LLM
chatmodel = ChatGroq(model="llama-3.1-8b-instant", temperature=0.15)

# ## setting up -> streamlit session state
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# if "memory" not in st.session_state:
#     st.session_state['memory'] = ConversationSummaryMemory(llm=ChatGroq(model="gemma-7b-it", temperature=0.2))

# ## resetting the entire conversation
# def reset_conversation():
#     st.session_state['messages'] = []
#     st.session_state['memory'].clear()

## cohere chat model
llm = ChatCohere(
    temperature=0.15
)

## open-source embedding model from HuggingFace - taking the default model only
embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## loading the vector database from local
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)

## setting up the retriever
kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 3})


"""
    HISTORY AWARE RETRIEVER
"""

contextualize_q_system_prompt = (
    """
        TASK: Convert context-dependent questions into standalone queries.

        INPUT: 
        - chat_history: Previous messages
        - question: Current user query

        RULES:
        1. Replace pronouns (it/they/this) with specific referents
        2. Expand contextual phrases ("the above", "previous")
        3. Return original if already standalone
        4. NEVER answer or explain - only reformulate

        OUTPUT: Single reformulated question, preserving original intent and style.

        Example:
        History: "Let's discuss Python."
        Question: "How do I use it?"
        Returns: "How do I use Python?"
    """
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm = llm,
    retriever = kb_retriever,
    prompt = contextualize_q_prompt
)

"""
    DOCUMENT CHAIN
"""


## setting-up the prompt
system_prompt_template = (
    "<s>[INST] As a Legal Assistant Chatbot specializing in legal queries, "
    "your primary objective is to provide accurate and concise information based on user queries. "
    "You will adhere strictly to the instructions provided, offering relevant "
    "context from the knowledge base while avoiding unnecessary details. "
    "Your responses will be brief, to the point, concise and in compliance with the established format. "
    "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
    "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
    "Use four sentences maximum. "
    "\nCONTEXT: {context}\nQUESTION: {question}"
    "\nANSWER:\n</s>[INST]"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


"""
RETRIEVAL CHAIN
"""
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


"""
    CONVERSATION SET-UP
"""


chat_history = []

question = "My friend Anusmita is a lesbian and she loves Subhasree. Can they get officially married?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
st.write("User: ", question)
st.write("AI: ", ai_msg_1["answer"])
chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "Can they have kids?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

st.write("User: ", second_question)
st.write("AI: ", ai_msg_2["answer"])