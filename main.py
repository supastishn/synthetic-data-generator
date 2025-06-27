#!/usr/bin/env python3

from litellm import completion
import os
from dotenv import load_dotenv
import xml.etree.ElementTree as ET  # Add this for XML parsing
import re  # Add for XML fragment extraction
import json  # Add for JSON serialization
import asyncio
import concurrent.futures
import time
import random
import litellm

def unescape_newlines(s):
    if not s:
        return s
    return s.replace('\\n', '\n').replace('\\t', '\t')

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

def is_retryable_exception(e):
    # Built-in network/timeout errors
    if isinstance(e, (ConnectionError, TimeoutError)):
        return True
        
    # LiteLLM specific exceptions
    if isinstance(e, (litellm.RateLimitError, litellm.Timeout, 
                      litellm.APITimeoutError, litellm.ServiceUnavailableError)):
        return True
        
    # Check by status code
    code = getattr(e, 'status_code', None)
    if code in [429, 500, 502, 503, 504]:
        return True
    
    return False

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

# At the top with other env variables
prompt_instructions = unescape_newlines(
    get_env_or_prompt(
        "PROMPT_INSTRUCT",
        "Enter custom instructions for the prompt generation model (optional): ",
        ""
    )
)

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

append_mode = get_env_or_prompt(
    "APPEND",
    "Append to output file if exists? (y/N): ",
    "n"
).lower() == "y"

existing_conversations = []
if append_mode and os.path.exists(output_file):
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_conversations = json.load(f)
        print(f"Loaded {len(existing_conversations)} existing conversations from {output_file}.")
    except Exception as e:
        print(f"WARNING: Failed to load existing file {output_file}: {e}. Starting new file.")

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
gen_batch_size = get_env_or_prompt(
    "GEN_BATCH_SIZE",
    "Enter batch size for prompt generation (default 5): ",
    "5"
)
try:
    gen_batch_size = int(gen_batch_size)
except ValueError:
    print("Invalid batch size. Using default 5")
    gen_batch_size = 5

# Max threads for async operations
max_workers = os.getenv("MAX_THREADS", "10")
try:
    max_workers = int(max_workers)
except ValueError:
    print("Invalid MAX_THREADS. Using default 10")
    max_workers = 10

# Async generation flag
async_gen = get_env_or_prompt(
    "ASYNC_GEN",
    "Enable simultaneous generation? (y/N): ",
    "n"
).lower() == "y"

if len(amounts) == 1:
    amounts = [amounts[0]] * len(topics)

def generate_prompts(topic = "Any", amount = 1, prompt_instructions=""):
    if verbose_logging:
        print(f"\n{'='*40}\nGenerating {amount} prompts for: {topic}\n{'='*40}")

    user_message = f"""
    Generate exactly {amount} prompts for '{topic}'."""

    # Add the instructions block if provided
    if prompt_instructions:
        user_message += f"""

    Additional Instructions:
    {prompt_instructions}"""

    user_message += """

    Format requirements:
    1. Each prompt must be wrapped in <prompt> tags 
    2. Each prompt must contain exactly two tags: 
       - A <system> tag with a system prompt
       - A <user> tag with a user prompt
    3. Output ONLY the XML-formatted prompts with no additional text
    4. Example format: 
       <prompt>
          <system>System instruction for AI</system>
          <user>User question</user>
       </prompt>
    5. For multiple prompts, output them consecutively without separators   
    """

    max_retries = 3  # Maximum retry attempts
    base_delay = 1    # Base delay in seconds
    attempt = 0

    while True:
        try:
            if verbose_logging:
                log_request(promptgen_model, [
                    {"content": "You output only in XML format. Use <prompt>, <system>, and <user> tags. Do not include any explanations or additional text.", "role": "system"},
                    {"content": user_message, "role": "user"}
                ], temperature=0.7)

            response = completion(
                model=promptgen_model,
                messages=[
                    {  # More strict system message
                        "content": "You output only in XML format. Use <prompt>, <system>, and <user> tags. Do not include any explanations or additional text.",
                        "role": "system"
                    },
                    {"content": user_message, "role": "user"}
                ],
                temperature=0.7,
            )

            if verbose_logging:
                log_response(response, promptgen_model)
            break  # Exit loop on success
        except Exception as e:
            if attempt < max_retries and is_retryable_exception(e):
                sleep_time = base_delay * (2 ** attempt) * (1 + random.random() * 0.1)  # Exponential backoff with jitter
                print(f"⚠️ Generation failed: {str(e)}. Retry {attempt+1}/{max_retries} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                attempt += 1
            else:
                print(f"❌ Failed after {attempt} retries: {e}")
                raise

    # Extract and parse XML content
    xml_resp = response.choices[0].message.content

    # Extract XML-like prompt tags ignoring surrounding text
    prompt_parts = re.findall(r'(<prompt>.*?</prompt>)', xml_resp, re.DOTALL)
    if not prompt_parts:
        print(f"No valid <prompt> tags found for topic '{topic}'")
        print(f"Received response: {xml_resp}")
        exit(1)

    result = []
    # Parse each prompt block with regex to allow content with angle brackets
    for xml_block in prompt_parts:
        # Extract system content
        system_match = re.search(r'<system>(.*?)</system>', xml_block, re.DOTALL)
        if not system_match:
            print(f"ERROR: Could not find <system> tags in prompt for topic '{topic}'")
            print(f"XML Block: {xml_block}")
            exit(1)
        system_content = system_match.group(1).strip()

        # Extract user content
        user_match = re.search(r'<user>(.*?)</user>', xml_block, re.DOTALL)
        if not user_match:
            print(f"ERROR: Could not find <user> tags in prompt for topic '{topic}'")
            print(f"XML Block: {xml_block}")
            exit(1)
        user_content = user_match.group(1).strip()

        result.append([
            {"role": "system", "content": system_content, "generation_model": promptgen_model},
            {"role": "user", "content": user_content, "generation_model": promptgen_model}
        ])
    return result



def generate_answers(messages, model_to_use, logits=False):
    response_options = {}
    if logits:
        response_options["logprobs"] = True
        response_options["top_logprobs"] = 10

    max_retries = 3   # Maximum retry attempts
    base_delay = 1    # Base delay in seconds
    attempt = 0

    while True:
        try:
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
            break  # Exit loop on success
        except Exception as e:
            if attempt < max_retries and is_retryable_exception(e):
                sleep_time = base_delay * (2 ** attempt) * (1 + random.random() * 0.1)  # Exponential backoff with jitter
                print(f"⚠️ Generation failed: {str(e)}. Retry {attempt+1}/{max_retries} in {sleep_time:.1f}s")
                time.sleep(sleep_time)
                attempt += 1
            else:
                print(f"❌ Failed after {attempt} retries: {e}")
                raise

    # Extract the core response content
    assistant_content = response.choices[0].message.content.strip()
    
    # Check for optional reasoning/thinking data
    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
    thinking_blocks = getattr(response.choices[0].message, 'thinking_blocks', [])
    
    reasoning_text = ""
    if reasoning_content:
        reasoning_text = reasoning_content.strip()
    elif thinking_blocks:
        # Combine all thinking blocks into one string
        reasoning_text = "\n\n".join(
            block.get("thinking", "").strip()
            for block in thinking_blocks
            if block.get("type") == "thinking" and block.get("thinking")
        )
    
    # Prepend reasoning if present
    if reasoning_text:
        assistant_content = f"<think>\n{reasoning_text}\n</think>\n\n{assistant_content}"
    
    result_message = {
        "role": "assistant",
        "content": assistant_content,
        "generation_model": model_to_use
    }

    if logits:
        result_message["logprobs"] = response.choices[0].logprobs

    messages.append(result_message)
    return messages

# --- Async support functions ---

async def generate_prompts_async(topic, amount, batch_size, prompt_instructions=""):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    batches = (amount + batch_size - 1) // batch_size
    tasks = []
    remaining = amount

    for _ in range(batches):
        task_amount = min(batch_size, remaining)
        remaining -= task_amount
        tasks.append(
            asyncio.get_event_loop().run_in_executor(
                pool, generate_prompts, topic, task_amount, prompt_instructions
            )
        )

    return await asyncio.gather(*tasks)

async def generate_answers_async(messages_list, models_list, logits):
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    tasks = []

    for msg, model in zip(messages_list, models_list):
        tasks.append(
            asyncio.get_event_loop().run_in_executor(
                pool, generate_answers, msg, model, logits
            )
        )

    return await asyncio.gather(*tasks)

conversations = existing_conversations[:]  # Start with existing conversations
new_conversations_indices = list(range(len(conversations), len(conversations) + sum(amounts)))

# Prompt generation (sync or async)
if async_gen:
    # Process topics sequentially with async batching within each topic
    loop = asyncio.get_event_loop()
    for topic_index, current_topic in enumerate(topics):
        amount_for_topic = amounts[topic_index]
        batches = loop.run_until_complete(
            generate_prompts_async(current_topic, amount_for_topic, gen_batch_size, prompt_instructions)
        )
        for batch in batches:
            for group in batch:
                if len(conversations) not in new_conversations_indices:
                    continue  # Skip non-new indices
                conversations.append({"messages": group})
else:
    # Sync prompt generation (now batched like async mode)
    for topic_index, current_topic in enumerate(topics):
        amount_for_topic = amounts[topic_index]
        
        # Calculate batches for sync mode
        batches = []
        remaining = amount_for_topic
        while remaining > 0:
            batch_amount = min(remaining, gen_batch_size)
            batches.append(batch_amount)
            remaining -= batch_amount
            
        for batch_amount in batches:
            prompt_groups = generate_prompts(current_topic, batch_amount, prompt_instructions)
            for group in prompt_groups:
                if len(conversations) not in new_conversations_indices:
                    continue  # Skip non-new indices
                conversations.append({"messages": group})

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
        messages_list = []
        for idx, conv in enumerate(conversations):
            if idx not in new_conversations_indices:
                continue  # Skip existing conversations
            messages_list.append(conv["messages"])
        updated_messages = await generate_answers_async(
            messages_list, assigned_models[-len(messages_list):], logits
        )
        update_idx = 0
        for idx, conv in enumerate(conversations):
            if idx not in new_conversations_indices:
                continue
            conv["messages"] = updated_messages[update_idx]
            conv["model"] = assigned_models[idx]
            update_idx += 1
    asyncio.get_event_loop().run_until_complete(gather_answers())
else:
    for idx, conversation in enumerate(conversations):
        if idx not in new_conversations_indices:
            continue  # Skip existing conversations
        model_to_use = assigned_models[idx]
        conversation['messages'] = generate_answers(
            conversation['messages'],
            model_to_use,
            logits
        )
        conversation['model'] = model_to_use

# Add at the very end of the script, after processing conversations
print(f"\nSaved {len(new_conversations_indices)} new conversations (total: {len(conversations)}) to {output_file}")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2, cls=EnhancedJSONEncoder)
