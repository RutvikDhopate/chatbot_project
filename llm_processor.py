import os
import uuid
from typing import List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from llm_sandbox import SandboxSession
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Load ENV Varialbe
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define Graph State
class GraphState(TypedDict):
    error: str
    messages: List
    generation: Dict
    iterations: int
    dataset_path: str

# Data Model for Structured Output
class CodeOutput(BaseModel):
    prefix: str = Field(description="Description of approach to the problem")

# Create the Prompt from the Text File
def format_prompt(prompt_file):
    prompt = ""
    with open(prompt_file) as file:
        for line in file:
            prompt += line
    return prompt

system_prompt = format_prompt("llm_prompt.txt")

# Prompt Template
output_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{system_prompt}" )
])   

def processor():
    return