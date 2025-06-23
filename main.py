from litellm import completion
import os
import xml.etree.ElementTree as ET  # Add this for XML parsing
import re  # Add for XML fragment extraction

# Check for necesarry lenvironment variable

if "PROMPTGEN_MODEL" not in os.environ:
    print("Error: The environment variable PROMOTGEN_MODEL is not set.")
    exit(1)

if "ANSWERGEN_MODEL" not in os.environ:
    print("Error: The environment variable ANSWERGEN_MODEL is not set.")
    exit(1)

if "TEMPERATURE" not in os.environ:
    print("Warning: The environment variable TEMPERATURE is not set. Using default value of 0.7.")
    temp = 0.7

promptgen_model = os.environ["PROMPTGEN_MODEL"]
answergen_model = os.environ["ANSWERGEN_MODEL"]


topic = input("Please enter 1 or more topics, separated by a comma: ")

topics = [topic.strip() for topic in topic.split(",")]

amount = input("How many prompt/answer combination do you want to create? Please enter either one number, or the same amount of numbers as topics, sepatated by a comma: ")

amounts = [int(a.strip()) for a in amount.split(",")]

if len(amounts) != 1 and len(amounts) != len(topics):
    print("Error: The amount of numbers must either be 1 or the same amount as topics.")
    exit(1)

multiprompt_input = ""

while multiprompt_input not in ["y", "yes", "n", "no", ""]:
    multiprompt_input = input("Do you want to use multiprompt generation? This saves requests by generating 10 prompts per request. (Y/n): ").strip().lower()
    if multiprompt_input not in ["y", "yes", "n", "no", ""]:
        print("Invalid input. Please enter 'y' or 'n'.")

multiprompt = False

if multiprompt_input == "y":
    multiprompt = True


if len(amounts) == 1:
    amounts = [amounts[0]] * len(topics)

prompts = []

def generate_prompt(topic, amount):
    print(f"Generating {amount} prompts for topic: {topic}")
    # Enhanced XML instructions
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

    try:
        root = ET.fromstring(xml_clean)
        for elem in root.findall('prompt'):
            if elem.text and elem.text.strip():
                prompts.append(elem.text.strip())
    except ET.ParseError as e:
        print(f"XML Parsing failed for topic '{topic}': {e}")
        print(f"Cleaned XML block: {xml_clean}")
        exit(1)


