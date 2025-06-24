import os
import pandas as pd
import torch
import torch.nn as nn
import json
import transformers
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel

model_name = "meta-llama/Llama-2-7b-hf"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,  # Adjust as needed
    dtype=torch.float16,  # Or torch.bfloat16
    load_in_4bit=True,
)

tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,  # Optional: for longer sequences
    random_state=3407,
    max_seq_length=2048,
)

def convert_conversation_to_pair(messages):
    """Convert conversation messages to human/assistant pair"""
    human = ""
    assistant = ""
    for msg in messages:
        if msg["role"] == "user":
            human = msg["content"]
        elif msg["role"] == "assistant":
            assistant = msg["content"]
    return human, assistant

# Generate prompts for the dataset  
def generate_prompt(data_point):
  return f"""
<Human>: {data_point['human']}
<AI>: {data_point['assistant']}
  """.strip()


def generate_and_tokenize_prompt(data_point):
  full_prompt = generate_prompt(data_point)
  tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
  return tokenized_full_prompt

# Load local conversations.json file
with open('conversations.json', 'r') as f:
    conversations_data = json.load(f)

# Create pairs from conversations
data_pairs = []
for conversation in conversations_data:
    human_msg, assistant_msg = convert_conversation_to_pair(conversation["messages"])
    if human_msg and assistant_msg:  # Only include valid pairs
        data_pairs.append({"human": human_msg, "assistant": assistant_msg})

# Create dataset from the pairs
dataset = Dataset.from_list(data_pairs)

# Tokenize dataset
dataset = dataset.shuffle().map(generate_and_tokenize_prompt)
