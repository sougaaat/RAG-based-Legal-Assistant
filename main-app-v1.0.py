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

## setting up -> streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# resetting the entire conversation
def reset_conversation():
    st.session_state['messages'].clear()

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

## initiating the history_aware_retriever
rephrasing_template = (
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

rephrasing_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rephrasing_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm = llm,
    retriever = kb_retriever,
    prompt = rephrasing_prompt
)


## setting-up the document chain
system_prompt_template = (
    "As a Legal Assistant Chatbot specializing in legal queries, "
    "your primary objective is to provide accurate and concise information based on user queries. "
    "You will adhere strictly to the instructions provided, offering relevant "
    "context from the knowledge base while avoiding unnecessary details. "
    "Your responses will be brief, to the point, concise and in compliance with the established format. "
    "If a question falls outside the given context, you will simply output that you are sorry and you don't know about this. "
    "The aim is to deliver professional, precise, and contextually relevant information pertaining to the context. "
    "Use four sentences maximum."
    "\nCONTEXT: {context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
## final RAG chain
coversational_rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

## setting-up conversational UI

## printing all (if any) messages in the session_session `message` key
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.write(message.content)

user_query = st.chat_input("Ask me anything ..")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            ## invoking the chain to fetch the result
            result = coversational_rag_chain.invoke({"input": user_query, "chat_history": reset_conversation})

            message_placeholder = st.empty()

            full_response = (
                "⚠️ **_This information is not intended as a substitute for legal advice. "
                "We recommend consulting with an attorney for a more comprehensive and"
                " tailored response._** \n\n\n"
            )
        
        ## displaying the output on the dashboard
        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02) ## <- simulate the output feeling of ChatGPT

            message_placeholder.markdown(full_response + " ▌")
        st.button('Reset Conversation 🗑️', on_click=st.session_state['messages'].clear())
    ## appending conversation turns
    st.session_state.messages.extend(
        [
            HumanMessage(content=user_query),
            AIMessage(content=result['answer'])
        ]
    )