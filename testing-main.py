## dashboard
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.session_state.chat_history.extend(
    [
        HumanMessage(content="Who is Adam Levine?"),
        AIMessage(content="Lead Singer of Maroon5.")
    ]
)

st.write(st.session_state)


st.session_state.chat_history.extend(
    [
        HumanMessage(content="Who's Sumner Stroh?"),
        AIMessage(content="His girlfriend.")
    ]
)

if st.button(label="RESET"):
    st.session_state.chat_history.clear()

st.write(st.session_state)