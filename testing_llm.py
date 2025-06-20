import json
import os
import glob
from datetime import datetime
from openai_llm import ChatBot

INPUT_DIR = "input_jsons"
OUTPUT_DIR = "output_jsons"

# Example input JSON format:
# {"messages": [
#   {"type": "text", "content": "Hello!"},
#   {"type": "image", "content": "data:image/png;base64,iVBORw0KGgoAAAANS..."},
#   {"type": "text", "content": "What is in this image?"}
# ]}

def process_file(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    user_messages = data.get("messages", [])

    provider = "groq"
    model = "meta-llama/llama-4-scout-17b-16e-instruct"
    bot = ChatBot(provider, model)

    conversation = []
    for msg in user_messages:
        if isinstance(msg, dict):
            content_type = msg.get("type", "text")
            content = msg.get("content", "")
        else:
            content_type = "text"
            content = msg
        bot.add_user_message(content, content_type)
        reply = bot.generate()
        # Truncate image content in output
        if content_type == "image":
            display_content = content[:30] + "...<truncated>..." if len(content) > 40 else content
        else:
            display_content = content
        conversation.append({"user": {"type": content_type, "content": display_content}, "assistant": reply})

    # Prepare output filename
    base = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base}_{timestamp}.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, "w") as f:
        json.dump({"conversation": conversation}, f, indent=2)
    print(f"Conversation saved to {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not input_files:
        print(f"No input files found in {INPUT_DIR}")
        return
    for input_path in input_files:
        process_file(input_path)

if __name__ == "__main__":
    main()

# Sample input JSON for text and image:
# {
#   "messages": [
#     {"type": "text", "content": "Hello!"},
#     {"type": "image", "content": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."},
#     {"type": "text", "content": "What is in this image?"}
#   ]
# }
