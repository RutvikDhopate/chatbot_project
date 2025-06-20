import os
import pandas as pd
import streamlit as st
import json
from dotenv import load_dotenv
from openai_llm import ChatBot
from utils import process_file_type, preview_file

# Load ENV Variables
load_dotenv()

st.set_page_config(page_title="Customer Support ChatBot", layout="wide")

# Session State Initialization
if 'conversation' not in st.session_state: st.session_state.conversation = []
if 'model_choice' not in st.session_state: st.session_state.model_choice = "groq-llama"

# Title and Description
st.title("Customer Support ChatBot")
st.markdown("""
This is a domain-specific ChatBot used to help users understand any issues related to swim channels.
""")

# Sidebar - Configuration
with st.sidebar:
    st.header("Configuration")

    # API Key Setup
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    st.subheader("Select Model")
    model_options = [
        "groq-gemma",
        "groq-llama",
        "openai-gpt4.1",
        "openai-o4mini",
        "gemini-2.5"
    ]

    model_choice = st.selectbox(
        "Choose an LLM Model",
        model_options,
        index=model_options.index(st.session_state.model_choice),
        key="model_selector"
    )
    st.session_state.model_choice = model_choice

    # Map model to provider and internal model name
    MODEL_PROVIDER_MAP = {
        "groq-gemma": ("groq", "llama-3.3-70b-versatile"),
        "groq-llama": ("groq", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "openai-gpt4.1": ("openai", "gpt-4"),
        "openai-o4mini": ("openai", "gpt-3.5-turbo"),
        "gemini-2.5": ("gemini", "gemini-pro")
    }

    # API Key Connection Status
    selected_provider, _ = MODEL_PROVIDER_MAP[model_choice]
    key_var = {
        "groq": groq_api_key,
        "openai": openai_api_key,
        "gemini": gemini_api_key
    }.get(selected_provider)

    if key_var:
        st.success(f"{model_choice} API Key: ✓ Connected")
    else:
        st.error(f"{model_choice} API Key: ✗ Missing")
        st.info(f"Add the API key for `{model_choice}` to your .env file.")

    # Dataset Upload
    st.subheader("Upload Data")
    uploaded = st.file_uploader("Choose a CSV/Excel/JSON/Image File",
                                type=["png", "jpg", "jpeg", "json", "csv", "xlsx", "pdf"],
                                accept_multiple_files=True)
    
    processed_files = []
    for file in uploaded:
        file_type = file.type
        file_text = process_file_type(file=file)
        processed_files.append((file, file_text))

    # Disclaimer Information
    st.info("Please note that if you click x on the uploaded file, it is still passed to the LLM as context.\n"
    "Please refresh the screen to start over.\n"
    "Alternatively, please hit Clear Conversation to clear LLM's context")

    # Clear conversation button
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        if 'assistant' in st.session_state:
            del st.session_state.assistant
        st.success("Conversation cleared.")

# Instantiate ChatBot
provider, model = MODEL_PROVIDER_MAP[st.session_state.model_choice]
if 'assistant' not in st.session_state: st.session_state.assistant = ChatBot(provider=provider, model=model)

# Reinitialize assistant if model/provider changed
else:
    if (
    st.session_state.assistant.model != model or
    st.session_state.assistant.provider != provider
    ):
        old_messages = st.session_state.assistant.messages if 'assistant' in st.session_state else []
        st.session_state.assistant = ChatBot(provider=provider, model=model)
        for msg in old_messages:
            if msg["role"] == "user":
                st.session_state.assistant.add_user_message(msg["content"])
            elif msg["role"] == "assistant":
                st.session_state.assistant.add_bot_message(msg["content"])



# Layout - Preview + Conversation
col1, col2 = st.columns([2.5, 2.5])

with col1:
    st.subheader("Dataset Preview")
    for file, file_text in processed_files:
        preview_file(file)

        # Process File Text to Add it to LLM's messages
        if "image" in file.type:
            st.session_state.assistant.add_user_message(file_text, content_type="image")
        else:
            file_text = process_file_type(file)
            st.session_state.assistant.add_user_message(file_text, content_type="text")


with col2:
    st.subheader("Conversation")

    # Get user input first
    ui = st.chat_input("How can I assist you?")
    if ui:
        st.session_state.assistant.add_user_message(ui)
        st.session_state.conversation.append({"role": "user", "content": ui})
        # st.session_state.conversation.append({"role": "user", "content": ui})
        with st.spinner(f"Generating response via {st.session_state.model_choice}..."):
            try:
                out = st.session_state.assistant.generate()
                st.session_state.conversation.append({"role": "assistant", "content": {"prefix": out}})
            except Exception as e:
                st.error(f"Error generating output: {e}")

    # Render full conversation
    for message in reversed(st.session_state.conversation):
        with st.chat_message(message["role"]):
            st.markdown(message["content"] if message["role"] == "user" else message["content"]["prefix"])