import os
import uuid
from typing import List, Dict
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_groq.chat_models import ChatGroq
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from utils import format_prompt

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

# Create the Prompt from the Text File
system_prompt = format_prompt("llm_prompt.txt")

# Prompt Template
output_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{system_prompt}" )
])   

# LangGraph Nodes Helper Functions
def generate(state: GraphState, output_chain):
    try:
        print(f"Generation - {state['iterations']}")
        messages = []
        for role, content in state["messages"]:
            if not content:
                messages.append("Hi")
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        solution = output_chain.invoke(messages)
    except Exception as e:
        state["error"] =  "yes"
        state["iterations"] += 1
        print(f"Error during output generation: {e}")
        return state
    
    if solution:
        state["messages"].append(("assistant", solution))
        state["generation"] = {"output": solution}
        state["error"] = "no"

    state["iterations"] += 1
    return state

def decide_finish(state: GraphState) -> str:
    if state['error'] == "no":
        return 'end'
    elif state['error'] == "yes" and state['iterations'] < 3:
        return 'generate'
    else:
        return 'end'

# Retrieve the Appropriate LLM Based on User's Model Choice
def retrieve_llm(model_choice="groq-llama"):
    if model_choice == "groq-llama":
        return ChatGroq(temperature=0.1, model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    elif model_choice == "groq-gemma":
        return ChatGroq(temperature=0.1, model="gemma2-9b-it", api_key=GROQ_API_KEY)
    elif model_choice == "openai-gpt4.1":
        return ChatOpenAI(temperature=0.1, model="gpt-4.1", api_key=OPENAI_API_KEY)
    elif model_choice == "openai-04mini":
        return ChatOpenAI(model="o4-mini", api_key=OPENAI_API_KEY)
    elif model_choice == "gemini-2.5":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", api_key=GEMINI_API_KEY)
    else:
        # Default to Groq Llama
        return ChatGroq(temperature=0.1, model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


def processor(user_query: str, dataset_info: Dict, dataset_path: str, model_choice="groq-llama") -> Dict:
    # Build Dataset Information into Text
    info = f"Dataset name: {dataset_info['name']}\n"
    info += f"Shape: {dataset_info['shape'][0]} rows, {dataset_info['shape'][1]} columns\n"
    info += "Columns:\n"
    for col in dataset_info['columns']:
        info += f"- {col['name']} ({col['type']})\n"
    info += f"\nSample data:\n{dataset_info['sample']}"

    # Initialize LLM Chain based on model choice
    llm = retrieve_llm(model_choice=model_choice)
    code_gen_chain = llm

    # Create custom generate function that includes the node chain
    def generate_with_chain(state):
        if user_query != "":
            return generate(state, code_gen_chain)

    # Build workflow graph
    workflow = StateGraph(GraphState)
    workflow.add_node("generate", generate_with_chain)
    workflow.add_edge(START, "generate")
    workflow.add_conditional_edges("generate", decide_finish, {"end": END, "generate": "generate"})
    thread_cfg = {"configurable": {"thread_id": uuid.uuid4()}}
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # Initialize Graph State
    initial_state: GraphState = {
        "messages": [
            ("system", f"""{system_prompt}\n\nDataset Info{info}"""),
            ("user", user_query)
        ],
        "iterations": 0,
        "error": "",
        "dataset_path": dataset_path,
        "generation": {}
    }

    # Invoke Graph
    final_state = graph.invoke(initial_state, config=thread_cfg)
    sol = final_state["generation"]
    return {'prefix': sol["output"].content if isinstance(sol["output"], AIMessage) else str(sol["output"])}