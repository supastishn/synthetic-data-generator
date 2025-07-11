#!/usr/bin/env python3

import torch
import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

# Load environment variables
model_name = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-Coder-1.5B-Instruct")
max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "2048"))  # Can be any value, we auto support RoPE Scaling
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = os.getenv("LOAD_IN_4BIT", "True") == "True"  # Use 4bit quantization to reduce memory usage. Can be False.

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Ensure padding on the right

# Apply Qwen 2.5 chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",
    map_eos_token=True,
)

# Configure LoRA
r = int(os.getenv("LORA_R", "16"))
target_modules_str = os.getenv("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
target_modules = [m.strip() for m in target_modules_str.split(",")]
print(f"Using LoRA target modules: {target_modules}")

model = FastLanguageModel.get_peft_model(
    model,
    r=r,
    target_modules=target_modules,
    lora_alpha=r*2,  # Rule of thumb: 2x Lora r
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Use optimized method
    random_state=3407,
    max_seq_length=max_seq_length,
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text" : texts}

data_file = os.getenv("DATA_FILE", "conversations.json")
dataset = load_dataset("json", data_files=data_file, split="train")

# Standardize to role/content format
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Set up trainer
batch_size = int(os.getenv("BATCH_SIZE", 1))
grad_acc_steps = int(os.getenv("GRAD_ACC_STEPS", 4))
learning_rate = float(os.getenv("LEARNING_RATE", 2e-4))
training_args = TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    warmup_steps=5,
    max_steps=30,  # Can set to None for full epoch training
    learning_rate=learning_rate,
    logging_steps=1,
    optim="paged_adamw_8bit",  # Save more memory
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=training_args,
    packing=False,  # Recommended for Qwen training
)

# Apply response-only masking for assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

# Start training
trainer.train()

# Save final model
model.save_pretrained_merged("unsloth_final_model", tokenizer, save_method="merged_4bit")
print("Model saved successfully!")
