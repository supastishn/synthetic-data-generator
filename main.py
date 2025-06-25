from litellm import completion
import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET  # Add this for XML parsing
import re  # Add for XML fragment extraction
import json  # Add for JSON serialization
import asyncio
import concurrent.futures

def log_request(model, messages, **kwargs):
    print(f"\n[VERBOSE] REQUEST to {model}:")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    if kwargs:
        print(f"Options: {json.dumps(kwargs, indent=2)}")

def log_response(response, model):
    content = response.choices[0].message.content
    logprobs = bool(getattr(response.choices[0].message, 'logprobs', None))
    
    print(f"\n[VERBOSE] RESPONSE from {model}:")
    print(f"Content: {content[:400]}{'...' if len(content) > 400 else ''}")
    if logprobs:
        print(f"Logprobs: captured ({len(response.choices[0].logprobs.content)} tokens)")

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, 'to_dict'):
            return o.to_dict()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        return super().default(o)

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

answergen_model = get_env_or_prompt("ANSWERGEN_MODEL", "Please enter the model name(s) for generating answers (comma-separated for multiple): ")
if not answergen_model:
    print("Error: ANSWERGEN_MODEL is required.")
    exit(1)

# Support multiple answer models
models = [m.strip() for m in answergen_model.split(",") if m.strip()]
if not models:
    print("Error: At least one answer model is required.")
    exit(1)

# Define a default split for multiple models
default_split = ",".join(["100"] * len(models))  # Single model case

model_split_str = get_env_or_prompt(
    "MODEL_SPLIT",
    f"Enter model split percentages (comma-separated, sum=100) for {len(models)} models: ",
    default=default_split
)

try:
    # Parse and validate split percentages
    splits = [int(s.strip()) for s in model_split_str.split(",")]
    if len(splits) != len(models):
        raise ValueError("Split count must match model count")
    if sum(splits) != 100:
        raise ValueError("Splits must sum to 100")
except Exception as e:
    print(f"Invalid MODEL_SPLIT: {e}")
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

verbose_logging = get_env_or_prompt(
    "VERBOSE_LOGGING",
    "Enable verbose logging for requests/responses? (y/N): ",
    "n"
).lower() == "y"

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

# Batch size for prompt/answer generation
batch_size = get_env_or_prompt(
    "BATCH_SIZE",
    "Enter batch size for prompt generation (default 5): ",
    "5"
)
try:
    batch_size = int(batch_size)
except ValueError:
    print("Invalid batch size. Using default 5")
    batch_size = 5

# Async generation flag
async_gen = get_env_or_prompt(
    "ASYNC_GEN",
    "Enable simultaneous generation? (y/N): ",
    "n"
).lower() == "y"

if len(amounts) == 1:
    amounts = [amounts[0]] * len(topics)

def generate_prompts(topic = "Any", amount = 1):
    if verbose_logging:
        print(f"\n{'='*40}\nGenerating {amount} prompts for: {topic}\n{'='*40}")

    user_message = f"""
    Generate exactly {amount} prompts for '{topic}'.
    Format requirements:
    1. Each prompt must be wrapped in <prompt> tags
    2. Output ONLY the XML-formatted prompts with no additional text
    3. Example format: <prompt>Your prompt here</prompt>
    4. For multiple prompts, output them consecutively without separators   
    """

    if verbose_logging:
        log_request(promptgen_model, [
            {"content": "You output only in XML format. Wrap all prompts in <prompt> tags. Do not include any explanations or additional text.", "role": "system"},
            {"content": user_message, "role": "user"}
        ], temperature=temp)

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

    if verbose_logging:
        log_response(response, promptgen_model)

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



def generate_answers(messages, model_to_use, logits=False):
    response_options = {}
    if logits:
        response_options["logprobs"] = True
        response_options["top_logprobs"] = 10

    if verbose_logging:
        log_request(model_to_use, messages, temperature=temp, **response_options)

    response = completion(
        model=model_to_use,
        messages=messages,
        temperature=temp,
        **response_options
    )

    if verbose_logging:
        log_response(response, model_to_use)

    assistant_response = response.choices[0].message.content.strip()

    result_message = {
        "role": "assistant",
        "content": assistant_response,
        "generation_model": model_to_use
    }

    if logits:
        result_message["logprobs"] = response.choices[0].logprobs

    messages.append(result_message)
    return messages

# --- Async support functions ---

async def generate_prompts_async(topic, amount, batch_size):
    pool = concurrent.futures.ThreadPoolExecutor()
    batches = (amount + batch_size - 1) // batch_size
    tasks = []
    remaining = amount

    for _ in range(batches):
        task_amount = min(batch_size, remaining)
        remaining -= task_amount
        tasks.append(
            asyncio.get_event_loop().run_in_executor(
                pool, generate_prompts, topic, task_amount
            )
        )

    return await asyncio.gather(*tasks)

async def generate_answers_async(messages_list, models_list, logits):
    pool = concurrent.futures.ThreadPoolExecutor()
    tasks = []

    for msg, model in zip(messages_list, models_list):
        tasks.append(
            asyncio.get_event_loop().run_in_executor(
                pool, generate_answers, msg, model, logits
            )
        )

    return await asyncio.gather(*tasks)

conversations = []

# Prompt generation (sync or async)
if async_gen:
    # Process topics sequentially with async batching within each topic
    conversations = []
    loop = asyncio.get_event_loop()
    for topic_index, current_topic in enumerate(topics):
        amount_for_topic = amounts[topic_index]
        batches = loop.run_until_complete(
            generate_prompts_async(current_topic, amount_for_topic, batch_size)
        )
        for batch in batches:
            for user_prompt in batch:
                user_prompt = dict(user_prompt)
                user_prompt["generation_model"] = promptgen_model
                conversations.append({"messages": [user_prompt]})
else:
    # Sync prompt generation (original code)
    for topic_index, current_topic in enumerate(topics):
        amount_for_topic = amounts[topic_index]
        user_prompts = generate_prompts(current_topic, amount_for_topic)
        for user_prompt in user_prompts:
            user_prompt = dict(user_prompt)
            user_prompt["generation_model"] = promptgen_model
            conversations.append({"messages": [user_prompt]})

# Assign answer models to conversations according to splits
assigned_models = []
for i, (model, split) in enumerate(zip(models, splits)):
    count = (split * len(conversations)) // 100
    if i == len(models) - 1:  # Last model gets remainder
        count = len(conversations) - len(assigned_models)
    assigned_models.extend([model] * count)

# Shuffle model assignments for fair distribution
import random
random.shuffle(assigned_models)

# Answer generation (sync or async)
if async_gen:
    async def gather_answers():
        messages_list = [conv["messages"] for conv in conversations]
        updated_messages = await generate_answers_async(
            messages_list, assigned_models, logits
        )
        for idx, (conv, updated) in enumerate(zip(conversations, updated_messages)):
            conv["messages"] = updated
            conv["model"] = assigned_models[idx]
    asyncio.get_event_loop().run_until_complete(gather_answers())
else:
    for idx, conversation in enumerate(conversations):
        model_to_use = assigned_models[idx]
        conversation['messages'] = generate_answers(
            conversation['messages'],
            model_to_use,
            logits
        )
        conversation['model'] = model_to_use

# Add at the very end of the script, after processing conversations
print(f"\nSaving {len(conversations)} conversations to {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2, cls=EnhancedJSONEncoder)
