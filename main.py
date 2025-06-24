from litellm import completion
import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET  # Add this for XML parsing
import re  # Add for XML fragment extraction
import json  # Add for JSON serialization

# Load environment variables from .env file
load_dotenv()

def get_env_or_prompt(env_var, prompt, default=None):
    value = os.getenv(env_var)
    if value is not None:
        return value.strip()
    user_input = input(prompt).strip()
    if user_input == "" and default is not None:
        return default
    return user_input

# Replace model handling
promptgen_model = get_env_or_prompt("PROMPTGEN_MODEL", "Please enter the model name for generating prompts: ")
if not promptgen_model:
    print("Error: PROMPTGEN_MODEL is required.")
    exit(1)

answergen_model = get_env_or_prompt("ANSWERGEN_MODEL", "Please enter the model name for generating answers: ")
if not answergen_model:
    print("Error: ANSWERGEN_MODEL is required.")
    exit(1)

# Replace TEMPERATURE handling
temp_str = get_env_or_prompt("TEMPERATURE", "Enter the temperature for generation (default 0.7): ", "0.7")
try:
    temp = float(temp_str)
except ValueError:
    print("Invalid format for temperature. Using default 0.7")
    temp = 0.7

# Add after other env variable assignments
output_file = get_env_or_prompt(
    "OUTPUT_FILE", 
    "Enter filename to save conversations (default conversations.json): ", 
    "conversations.json"
)

# Replace TOPICS handling
topic = get_env_or_prompt("TOPICS", "Please enter 1 or more topics, separated by a comma: ")
topics = [t.strip() for t in topic.split(",")]
if not topics or any(t == "" for t in topics):
    print("Error: At least one non-empty topic is required.")
    exit(1)

# Replace AMOUNTS handling
amount = get_env_or_prompt("AMOUNTS", "How many prompt/answer combinations? Enter one number or comma-separated numbers: ")
try:
    amounts = [int(a.strip()) for a in amount.split(",")]
except ValueError:
    print("Error: All amounts must be integers.")
    exit(1)
if len(amounts) != 1 and len(amounts) != len(topics):
    print("Error: Amount must be single number or same count as topics.")
    exit(1)

# Replace MULTI_PROMPT handling
multiprompt_input = get_env_or_prompt("MULTI_PROMPT", 
    "Use multiprompt generation? (Y/n): ", "y").lower()
multiprompt = True
if multiprompt_input == "n":
    multiprompt = False

# Replace LOGITS handling
logits_input = get_env_or_prompt("LOGITS", 
    "Use logits for answer generation? (y/N): ", "n").lower()
logits = False
if logits_input == "y":
    logits = True

if len(amounts) == 1:
    amounts = [amounts[0]] * len(topics)

def generate_prompts(topic = "Any", amount = 1):
    print(f"Generating {amount} prompts for topic: {topic} ")

    user_message = f"""
    Generate exactly {amount} prompts for '{topic}'.
    Format requirements:
    1. Each prompt must be wrapped in <prompt> tags
    2. Output ONLY the XML-formatted prompts with no additional text
    3. Example format: <prompt>Your prompt here</prompt>
    4. For multiple prompts, output them consecutively without separators   
    """
    
    response = completion(
        model=promptgen_model,
        messages=[
            {  # More strict system message
                "content": "You output only in XML format. Wrap all prompts in <prompt> tags. Do not include any explanations or additional text.",
                "role": "system"
            },
            {"content": user_message, "role": "user"}
        ],
        temperature=temp,
    )
    
    # Extract and parse XML content
    xml_resp = response.choices[0].message.content

    # Extract XML-like prompt tags ignoring surrounding text
    prompt_parts = re.findall(r'(<prompt>.*?</prompt>)', xml_resp, re.DOTALL)
    if not prompt_parts:
        print(f"No valid <prompt> tags found for topic '{topic}'")
        print(f"Received response: {xml_resp}")
        exit(1)

    # Build clean XML structure by wrapping elements
    xml_clean = f"<root>{''.join(prompt_parts)}</root>"

    result = []
    try:
        root = ET.fromstring(xml_clean)
        for elem in root.findall('prompt'):
            if elem.text and elem.text.strip():
                result.append({
                    "role": "user",
                    "content": elem.text.strip()
                })
        return result
    except ET.ParseError as e:
        print(f"XML Parsing failed for topic '{topic}': {e}")
        print(f"Cleaned XML block: {xml_clean}")
        exit(1)



def generate_answers(messages, logits=False):
    response = completion(
        model=answergen_model,
        messages=messages,
        temperature=temp,
        logits=logits
    )
    assistant_response = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": assistant_response})
    return messages

conversations = []

# Update loop variables and inner variable names
for topic_index, current_topic in enumerate(topics):
    amount_for_topic = amounts[topic_index]
    user_prompts = generate_prompts(current_topic, amount_for_topic)
    for user_prompt in user_prompts:
        conversations.append({
            "messages": [user_prompt]
        })

# Rename the conversation variable in the loop
for conversation in conversations:
    conversation['messages'] = generate_answers(conversation['messages'], logits=logits)

# Add at the very end of the script, after processing conversations
print(f"\nSaving {len(conversations)} conversations to {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)
