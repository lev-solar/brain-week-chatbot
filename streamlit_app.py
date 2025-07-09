import streamlit as st
from app import build_chain

st.set_page_config(page_title="Runbook Chatbot", layout="wide")
st.title("🧠 Runbook Chatbot (Local)")

qa_chain = build_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask a question about your runbooks...")
if user_query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(user_query)
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("bot", result))

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)