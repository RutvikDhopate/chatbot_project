import os
import pandas as pd
import streamlit as st
import json
from dotenv import load_dotenv
from llm_processor import processor
from utils import format_prompt, parse_image, parse_pdf


# Load ENV Variable
load_dotenv()

st.set_page_config(page_title="Customer Support ChatBot", layout="wide")

# Session State Initialization
if 'conversation' not in st.session_state: st.session_state.conversation = []
if 'dataset' not in st.session_state: st.session_state.dataset = None
if 'dataset_info' not in st.session_state: st.session_state.dataset_info = {}
if 'dataset_path' not in st.session_state: st.session_state.dataset_path = None
if 'model_choice' not in st.session_state: st.session_state.model_choice = "groq-llama"

# Title and Description
st.title("Customer Support ChatBot")
st.markdown("""
This is a domain specific ChatBot used to help
the users understand any issues related to any of the 
possible swim channels like Shipment, Order Status, Finance, 
Payments, etc. 
""")

# Sidebar - Choose AI Agent for Comparison
with st.sidebar:
    st.header("Configuration")

    # API Key Selection
    st.subheader("API Keys Setup")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if st.session_state.model_choice in ["groq-gemma", "groq-llama", "openai-gpt4.1", "openai-o4mini", "gemini-2.5"]:
        st.success(f"{st.session_state.model_choice} API Key:  ✓  Connected")
    else:
        st.error(f"{st.session_state.model_choice} API Key:  ✗  Missing")
        st.info(f"Add {st.session_state.model_choice} API Key to your ENV file. ")

    # Model Selection
    st.subheader("Select Model")
    model_choice = st.selectbox(
        "Choose an LLM Model",
        [
            "groq-gemma",
            "groq-llama",
            "openai-gpt4.1",
            "openai-o4mini",
            "gemini-2.5"
        ],
        index=["groq-gemma", "groq-llama", "openai-gpt4.1", "openai-o4mini", "gemini-2.5"].index(st.session_state.model_choice),
        key="model_selector"
    )

    # Update the session state when selection changes
    st.session_state.model_choice = model_choice

    # Data Upload
    st.subheader("Upload Data")
    combined_uploaded = st.file_uploader("Choose one or more CSV/Excel/JSON/Image File(s)", 
                                type=["csv", "xlsx", "xls", "json", "jpeg", "jpg", "png"],
                                accept_multiple_files=True)
    print(combined_uploaded)
    combined_df = pd.DataFrame()
    if combined_uploaded:
        for i, uploaded in enumerate(combined_uploaded):
            try:
                save_dir = os.path.join("data", "user_datasets")
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, uploaded.name)
                with open(path, 'wb') as file:
                    file.write(uploaded.getbuffer())
                if uploaded.name.endswith('csv'):
                    df = pd.read_csv(path)
                elif uploaded.name.endswith(('xls', 'xlsx')):
                    df = pd.read_excel(path)
                elif uploaded.name.endswith('json'):
                    df = pd.read_json(path)
                elif uploaded.name.endswith('pdf'):
                    pdf_text = parse_pdf(path)
                    # st.session_state.dataset_info = {
                    #     "name": uploaded.name,
                    #     "shape": (1, 1),
                    #     "columns": [{"name": "pdf_text", "type": "string", "description": "", "sample": pdf_text[:200]}],
                    #     "sample": pdf_text[:10000]
                    # }
                    # st.session_state.dataset = pd.DataFrame({"pdf_text": [pdf_text]})
                    df = pd.DataFrame({"source_file": [uploaded.name], "text": [pdf_text]})
                    st.success(f"{uploaded.name} loaded and parsed (PDF)")
                elif uploaded.name.endswith(("jpg", "jpeg", "png")):
                    img_text = parse_image(path)
                    # st.session_state.dataset_info = {
                    #     "name": uploaded.name,
                    #     "shape": (1, 1),
                    #     "columns": [{"name": "pdf_text", "type": "string", "description": "", "sample": img_text[:200]}],
                    #     "sample": img_text[:10000]
                    # }
                    # st.session_state.dataset = pd.DataFrame({"pdf_text": [img_text]})
                    df = pd.DataFrame({"source_file": [uploaded.name], "text": [img_text]})
                    st.success(f"{uploaded.name} loaded and parsed (Image)")
                else:
                    df = None
                
                if df is not None:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                    combined_df.reset_index(drop=True, inplace=True)
            except Exception as e:
                st.error(f"Erorr loading dataset {e}")

        if not combined_df.empty:
            combined_path = os.path.join(save_dir, "data.csv")
            combined_df.to_csv(combined_path, index=False)

            df = combined_df
        # df.to_csv(os.path.join(save_dir, 'data.csv'), index=False)
        # # Figure out image parsing to text
        # if df is not None:
            st.session_state.dataset = df
            st.session_state.dataset_path = os.path.join(save_dir, 'data.csv')
            st.session_state.dataset_info = {
                "name": uploaded.name,
                "shape": df.shape,
                "columns": [
                    {"name": c, "type": str(df[c].dtype), "description": "", "sample": str(df[c].iloc[0]) if not df.empty else ""} 
                    for c in df.columns
                ],
                "sample": df.head(50).to_string() + " " + df.tail(50).to_string()
            }   
            st.success(f"{uploaded.name} loaded correctly.")
            st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} cols")

    
    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.success("Conversation Cleared")


# Main Layout
col1, col2 = st.columns([3,2])
with col1:
    if st.session_state.dataset is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.dataset.head(), use_container_width=True)
        st.subheader("Conversation")

        
        ui = st.chat_input("How can I assist you?")
        if ui:
            # Add user message
            st.session_state.conversation.append({"role": "user", "content": ui, "type": "text"})

            # Generate output
            with st.spinner(f"Generating Output via {st.session_state.model_choice}..."):
                try:
                    out = processor(ui, st.session_state.dataset_info, st.session_state.dataset_path, st.session_state.model_choice)
                    st.session_state.conversation.append({"role": "bot", "content": {"prefix": out["prefix"]}, "type": "text"})
                    if not out["prefix"]:
                        st.error("No output received from the LLM.")
                except Exception as e:
                    st.error(f"Error Generating Output: {e}")

        for message in st.session_state.conversation:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["content"])
                elif message["role"] == "bot":
                    st.markdown(message["content"]["prefix"])