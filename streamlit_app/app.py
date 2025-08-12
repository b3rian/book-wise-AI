import streamlit as st
import httpx
import json
from typing import AsyncIterable, Awaitable

# Configuration
API_URL = "http://localhost:8000/prompt-stream"  # Update if your API is hosted elsewhere
DEFAULT_PROMPT = "What did Nietzsche say about morality?"
DEFAULT_N_RESULTS = 3

# App title and description
st.title("Nietzsche RAG Chat")
st.markdown("""
Ask questions about Nietzsche's philosophy and get answers powered by RAG.
Responses are streamed directly from the API.
""")

# Session state to store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
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
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            async for chunk in stream_response(prompt, n_results):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        except httpx.RequestError as e:
            st.error(f"Error connecting to API: {str(e)}")
            return
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Chat input
if prompt := st.chat_input("Ask your question about Nietzsche"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get number of results from sidebar
    n_results = st.sidebar.slider(
        "Number of context chunks",
        min_value=1,
        max_value=5,
        value=DEFAULT_N_RESULTS,
        help="How many document chunks to retrieve from the vector database"
    )
    
    # Display assistant response
    asyncio.run(display_stream_response(prompt, n_results))

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    st.markdown(f"API endpoint: `{API_URL}`")
    
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.experimental_rerun()