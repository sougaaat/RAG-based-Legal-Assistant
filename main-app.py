## setting up the environment
import time
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

## importing langchain dependencies
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

## importing dashboard -> streamlit
import streamlit as st

## designing the dashboard
st.set_page_config(page_title="RAG-Based Legal Assistant")
col1, col2, col3 = st.columns([1, 20, 1])
with col2:
    st.title("RAG-Based Legal Assistant")

def reset_conversation():
  st.session_state.messages = [] ## <- here `messages` is a key of the streamlit session_state. `messages` is also a python list.
  st.session_state.memory.clear() ## <- here `memory` is a key of the streamlit session_state, also `memory` is a list that's why .clear() method removes all the elements from the list

if "messages" not in st.session_state: ## <- checking if the key already exists
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history",return_messages=True)

embedF = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
knowledgeBase = FAISS.load_local("vector-store", embeddings=embedF, allow_dangerous_deserialization=True)
kbase_rtvr = knowledgeBase.as_retriever(search_type="similarity",search_kwargs={"k": 3})

promptTemplate = """<s>[INST] This is a chat template. As a Legal Assistant Chatbot specializing in legal queries,
your primary objective is to provide accurate and concise information based on user queries. 
You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. 
Your responses will be brief, to the point, concise and in compliance with the established format. 
If a question falls outside the given context, you will simply output that you are sorry and you don't know about this.
The aim is to deliver professional, precise, and contextually relevant information pertaining to the context.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=promptTemplate, input_variables=['context', 'question', 'chat_history'])

# tokenizer = AutoTokenizer.from_pretrained("model_name", clean_up_tokenization_spaces=True)
groq = ChatGroq(model="llama3-70b-8192", temperature=0.2, max_tokens=1024)

qa = ConversationalRetrievalChain.from_llm(
    llm=groq,
    memory=st.session_state.memory,
    retriever=kbase_rtvr,
    combine_docs_chain_kwargs={'prompt': prompt}
)

for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.chat_input("Ask me anything ..")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Generating 💡...", expanded=True):
            result = qa.invoke(input=input_prompt)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Disclaimer: This information is not a substitute for legal advice. Please consult with an attorney._** \n\n\n"
        for chunk in result["answer"]:
            full_response += chunk
            time.sleep(0.02)

            message_placeholder.markdown(full_response + " ▌")
        st.button('Reset All Chat 🗑️', on_click=reset_conversation)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})