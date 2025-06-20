import os
import json
import streamlit as st
import openai
from utils import format_prompt

class ChatBot:
    def __init__(self, provider, model):
        self.provider = provider
        self.model = model
        self.system_prompt = format_prompt("llm_prompt.txt")
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"))
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def add_user_message(self, content, content_type="text"):
        if content_type == "text":
            self.messages.append({"role": "user", "content": content})
        elif content_type == "image":
            self.messages.append({
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Please consider the content of the image to answer user's questions"},
                    {"type": "image_url", "image_url": {"url": content}}
                ]
            })
        
    def add_bot_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
    
    def generate(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.1
        )
        reply = response.choices[0].message.content
        self.add_bot_message(reply)
        return reply

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_messages(self):
        return self.messages
        