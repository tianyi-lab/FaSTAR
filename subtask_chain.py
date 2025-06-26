import json
import base64
import os
from pathlib import Path
from openai import OpenAI

def generate_subtask_chain(llm_api_key, image_path, prompt):

    client = OpenAI(api_key=llm_api_key)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"

    prompt_file_path = Path("prompts/subtask_chain_prompt.txt")
    if not prompt_file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    
    with open(prompt_file_path, "r") as f:
        prompt_template = f.read()
    
    message = prompt_template.replace("{prompt}", prompt)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message + "\n\nYou must respond in a valid JSON format. Do not include any extra text before or after the JSON output."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="o3",
        messages=messages
    )
    
    content = response.choices[0].message.content
    
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]

    try:
        subtask_chain = json.loads(content)

        if "task" not in subtask_chain or "subtask_chain" not in subtask_chain:
            raise ValueError("Invalid subtask chain structure: missing 'task' or 'subtask_chain' keys")

        start_nodes = [node for node in subtask_chain["subtask_chain"] if not node["parent"]]
        if len(start_nodes) != 1:
            raise ValueError(f"Invalid subtask chain structure: found {len(start_nodes)} start nodes, expected exactly 1")
        
        return subtask_chain
    except json.JSONDecodeError as e:
        print("Failed to parse response:")
        print("Raw content:", content)
        raise ValueError(f"Invalid JSON response: {str(e)}") from e

def main():
    llm_api_key = os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is not set")

    image_path = "path/to/your/image.jpg"  # Replace with your image path
    prompt = "Edit the image."             # Replace with your prompt

    try:
        subtask_chain = generate_subtask_chain(llm_api_key, image_path, prompt)
        print(json.dumps(subtask_chain, indent=2))
    except Exception as e:
        print(f"Error generating subtask chain: {e}")

# # Uncomment the lines below to run the main function when this script is executed
# if __name__ == "__main__":
#     main()
