#!/usr/bin/env python3

import os
import pandas as pd
import torch
import torch.nn as nn
import json
import transformers
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model_name = "meta-llama/Llama-2-7b-hf"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,  # Adjust as needed
    dtype=torch.float16,  # Or torch.bfloat16
    load_in_4bit=True,
)

tokenizer.pad_token = tokenizer.eos_token

# Add before model.get_peft_model
target_modules_str = os.getenv("TARGET_MODULES", "q_proj,v_proj")
target_modules = [m.strip() for m in target_modules_str.split(",")]
print(f"Using LoRA target modules: {target_modules}")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=target_modules,  # Updated
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
    return {"text": f"""
<Human>: {data_point['human']}
<AI>: {data_point['assistant']}
""".strip()}

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
dataset = Dataset.from_list(data_pairs).map(generate_prompt)

# Add SFTTrainer setup and training

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # Key for our generated prompts
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,            # Adjust based on GPU RAM
        gradient_accumulation_steps=4,             # 2x4=8 effective batch size
        learning_rate=2e-5,
        optim="paged_adamw_8bit",                 # For memory efficiency
        logging_steps=10,
        num_train_epochs=1,                       # Training epochs
        output_dir="unsloth_output",               # Save directory
        fp16=not torch.cuda.is_bf16_supported(),   # Precision settings
        bf16=torch.cuda.is_bf16_supported(),
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="epoch",
        evaluation_strategy="no"                   # No eval dataset
    ),
    packing=True                                   # Better sequence packing
)

# Start training
trainer.train()

# Save final model (add at end)
model.save_pretrained("unsloth_final_model")
print("Model saved successfully!")
