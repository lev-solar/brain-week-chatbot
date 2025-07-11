import streamlit as st
from app import build_chain

# Set up the page
st.set_page_config(page_title="Runbook Chatbot", layout="wide")
st.title("🧠 Runbook Chatbot (Local)")

# Step 1: Define available models
available_models = ["llama3.1", "llama4", "mistral"]

# Step 2: Persist model selection in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = available_models[0]  # default model

# Step 3: Show model selection dropdown
st.sidebar.subheader("🔧 Model Settings")
st.session_state.selected_model = st.sidebar.selectbox(
    "Select LLM model",
    available_models,
    index=available_models.index(st.session_state.selected_model)
)

# Step 4: Build the chain using selected model
@st.cache_resource(show_spinner=False)
def get_chain(model_name):
    return build_chain(model_name)

qa_chain = get_chain(st.session_state.selected_model)

# Step 5: Chat logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask a question about your runbooks...")

if user_query:
    with st.spinner("Thinking..."):
        result = qa_chain.run(user_query)
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("bot", result))

# Step 6: Display chat
for role, msg in st.session_state.chat_history:
    st.chat_message("user" if role == "user" else "assistant").write(msg)
