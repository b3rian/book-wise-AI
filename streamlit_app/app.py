import streamlit as st
import httpx
import json
from typing import AsyncIterable
import asyncio
from datetime import datetime, date, timedelta
import time

# Configuration
API_URL = "http://127.0.0.1:8000/prompt-stream"
DEFAULT_PROMPT = "What did Nietzsche say about morality?"
DEFAULT_N_RESULTS = 3

# Date formatting function
def format_conversation_date(created_at_str):
    # Parse the stored datetime string
    created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S").date()
    today = date.today()
    
    if created_at == today:
        return "Today"
    elif created_at == today - timedelta(days=1):
        return "Yesterday"
    elif created_at >= today - timedelta(days=6):  # Within last week
        return created_at.strftime("%A")  # Monday, Tuesday, etc.
    elif created_at.year == today.year:
        return created_at.strftime("%b %d")  # Aug 12
    else:
        return created_at.strftime("%Y-%m-%d")  # Fallback to full date

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = str(int(time.time()))
    st.session_state.conversations[st.session_state.current_conversation] = {
        "messages": [],
        "tags": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
if "typing_indicator" not in st.session_state:
    st.session_state.typing_indicator = False

# App title and description
st.markdown("---")
st.set_page_config(page_title="Zarathustra AI", layout="wide", page_icon="🖼️")
st.title("Zarathustra AI")
st.markdown("""
Ask questions about Nietzsche's philosophy and get answers powered by RAG.
Responses are streamed directly from the API.
""")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True) 
st.markdown("<br><br>", unsafe_allow_html=True) 

# Animated typing indicator
def show_typing_indicator():
    placeholder = st.empty()
    dots = ""
    while st.session_state.typing_indicator:
        dots = dots + "." if len(dots) < 3 else ""
        placeholder.markdown(f"Assistant is thinking{dots}")
        time.sleep(0.5)
    placeholder.empty()

# Display current conversation messages
current_messages = st.session_state.conversations[st.session_state.current_conversation]["messages"]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to stream API response
async def stream_response(prompt: str, n_results: int) -> AsyncIterable[str]:
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            API_URL,
            json={"prompt": prompt, "n_results": n_results},
            timeout=30.0
        ) as response:
            async for chunk in response.aiter_text():
                yield chunk

# Function to display streaming response
async def display_stream_response(prompt: str, n_results: int):
    st.session_state.typing_indicator = True
    asyncio.create_task(asyncio.to_thread(show_typing_indicator))
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            async for chunk in stream_response(prompt, n_results):
                if st.session_state.typing_indicator:
                    st.session_state.typing_indicator = False
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except httpx.RequestError as e:
            st.error(f"Error connecting to API: {str(e)}")
            return
        finally:
            st.session_state.typing_indicator = False
        
    # Add to conversation history
    current_conv = st.session_state.conversations[st.session_state.current_conversation]
    current_conv["messages"].extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_response}
    ])

# Conversation management functions
def new_conversation():
    conv_id = str(int(time.time()))
    st.session_state.current_conversation = conv_id
    st.session_state.conversations[conv_id] = {
        "messages": [],
        "tags": [],
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.rerun()

def load_conversation(conv_id):
    st.session_state.current_conversation = conv_id
    st.rerun()

def update_tags(conv_id, tags):
    st.session_state.conversations[conv_id]["tags"] = tags

# Sidebar - Conversation management
with st.sidebar:
    st.markdown("---")
    st.header("Chats")
    
    # New conversation button
    if st.button("+ New Chat"):
        new_conversation()
    
    # Tags for current conversation
    current_tags = st.text_input(
        "Tags for this conversation (comma separated)",
        value=", ".join(st.session_state.conversations[st.session_state.current_conversation]["tags"]),
        on_change=lambda: update_tags(
            st.session_state.current_conversation,
            [t.strip() for t in st.session_state.tags_input.split(",") if t.strip()]
        ),
        key="tags_input"
    )
    
    # Conversation history list with formatted dates
    st.markdown("---")
    st.subheader("Chat History")
    for conv_id, conv_data in sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    ):
        preview = ", ".join([msg["content"][:20] + "..." for msg in conv_data["messages"][:2] if msg["role"] == "user"])
        tags_display = " ".join([f"`{tag}`" for tag in conv_data["tags"]])
        friendly_date = format_conversation_date(conv_data["created_at"])
        
        if st.button(
            f"{friendly_date} - {preview} {tags_display}",
            key=f"conv_{conv_id}",
            use_container_width=True
        ):
            load_conversation(conv_id)
    
    # Configuration
    st.markdown("---")
    st.subheader("Configuration")
    n_results = st.slider(
        "Number of context chunks",
        min_value=1,
        max_value=5,
        value=DEFAULT_N_RESULTS,
        help="How many document chunks to retrieve from the vector database"
    )

# Quick preset prompts (above chat input)
preset_prompts = [
    "Summarize Beyond Good and Evil",
    "Explain Eternal Recurrence",
    "What is the Übermensch?",
    "What did Nietzsche say about morality?"
]

st.markdown("### Quick Prompts")
cols = st.columns(len(preset_prompts))
st.markdown("---")
for idx, preset in enumerate(preset_prompts):
    if cols[idx].button(preset):
        with st.chat_message("user"):
            st.markdown(preset)
        asyncio.run(display_stream_response(preset, n_results))

# Chat input
if prompt := st.chat_input("Ask your question about Nietzsche"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    asyncio.run(display_stream_response(prompt, n_results))