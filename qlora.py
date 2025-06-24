import os
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

# How to know what targt modules to use?
# You can check the model's architecture or documentation to find out which modules are suitable for LoRA.
# This script uses Qwen3's target modules as an example. 


lora_config = LoraConfig(
    r=16, # This is the rank of the LoRA layers.
    lora_alpha=32, # This is the scaling factor for the LoRA layers.
    target_modules=["q_proj", "v_proj"], # These are the target modules for LoRA.
    lora_dropout=0.05, # This is the dropout rate for the LoRA layers.
    bias="none", # This specifies how to handle bias in the LoRA layers.
    task_type="CAUSAL_LM", # This specifies the task type for the LoRA layers (e.g., CAUSAL_LM for causal language modeling).
)

model = get_peft_model(model, lora_config)


# Generate prompts for the dataset  
def generate_prompt(data_point):
  return f"""
<Human>: {data_point[0]}
<AI>: {data_point[1]}
  """.strip()

def generate_and_tokenize_prompt(data_point):
  full_prompt = generate_prompt(data_point)
  tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
  return tokenized_full_prompt

from datasets import load_dataset
dataset_name = 'Amod/mental_health_counseling_conversations'
dataset = load_dataset(dataset_name, split="train")

dataset = dataset.shuffle().map(generate_and_tokenize_prompt)
