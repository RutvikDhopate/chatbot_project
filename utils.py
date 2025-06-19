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
            pdf_text += pdf_text
    return pdf_text

# Parsing Text from Image
def parse_image(image):
    img = Image.open(image)
    img_text = pytesseract.image_to_string(img)
    return img_text