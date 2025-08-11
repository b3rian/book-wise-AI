import streamlit as st
import requests

# Your FastAPI backend endpoint
API_URL = "http://localhost:8000/query"  # Change this if deployed online

st.set_page_config(page_title="Philosopher Bot", page_icon="🤖", layout="centered")

st.title("🤖 Philosopher Bot")
st.write("Ask Kant, Nietzsche, Plato... and more!")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Philosopher Bot:** {msg['content']}")

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question:", placeholder="What is the meaning of life?")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        # Send to backend
        response = requests.post(API_URL, json={"query": user_input})
        if response.status_code == 200:
            bot_reply = response.json().get("answer", "No response from bot.")
        else:
            bot_reply = f"Error {response.status_code}: Could not reach API."

    except requests.exceptions.RequestException as e:
        bot_reply = f"Connection error: {e}"

    # Save bot reply
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.experimental_rerun()
