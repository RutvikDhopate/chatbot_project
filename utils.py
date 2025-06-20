import streamlit as st
import base64
import pandas as pd
import PyPDF2
import json
import fitz
import pytesseract
from PIL import Image

# Reading LLM System Prompt
def format_prompt(prompt_file):
    prompt = ""
    with open(prompt_file) as file:
        for line in file:
            prompt += line
    return prompt

# Parsing Text from PDF
def parse_pdf(pdf):
    pdf_text = ""
    with fitz.open(pdf) as file:
        for page in file:
            pdf_text += page
    return pdf_text

# Parsing Text from Image
def parse_image(image):
    img = Image.open(image)
    img_text = pytesseract.image_to_string(img)
    return img_text


def process_file_type(file):
    file.seek(0)
    # print(file.type)
    if "image" in file.type:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
        img_url = f"data:{file.type};base64,{base64_image}"
        return img_url
    
    elif "csv" in file.type:
        df = pd.read_csv(file)
        df.columns = df.columns.map(str)
        csv_text = df.to_csv(index=False)
        return csv_text

    elif "openxml" in file.type:
        df = pd.read_excel(file)
        df.columns = df.columns.map(str, )
        xlsx_text = df.to_csv(index=False)
        return xlsx_text

    elif file.name.endswith(".json"):
        json_text = file.read().decode("utf-8")
        return json_text

    elif file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        return pdf_text
    
    else:
        # Fallback: encode anything unknown as base64
        base64_generic = base64.b64encode(file.read()).decode("utf-8")
        return base64_generic
    

def preview_file(file):
    """Preview different file types in Streamlit based on extension and MIME type."""
    file.seek(0)
    st.markdown(f"#### Preview: `{file.name}`")

    # Image
    if "image" in file.type:
        st.image(file, caption=file.name)
    
    # CSV
    elif file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file)
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    # Excel
    elif file.name.endswith(".xlsx"):
        try:
            df = pd.read_excel(file)
            st.dataframe(df.head(10))
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")

    # JSON
    elif file.name.endswith(".json"):
        try:
            json_data = json.load(file)
            st.json(json_data)
        except Exception as e:
            st.error(f"Invalid JSON file: {e}")

    # PDF
    elif file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
            st.text_area("PDF Content", text, height=300)
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

    # Unknown type
    else:
        st.warning(f"No preview available for: {file.name}")

