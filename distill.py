#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch.nn.functional as F
from unsloth import FastLanguageModel

class DistillationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2048):
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.processed_samples = self._preprocess_data()
    
    def _get_full_logits(self, logprobs):
        """Convert token logprobs to full logits tensor"""
        logits = torch.full((self.tokenizer.vocab_size,), -10000.0)
        for logprob in logprobs:
            token = logprob['token']
            if not isinstance(token, bytes):
                token = token.encode('utf-8', 'backslashreplace')
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            logits[token_id] = logprob['logprob']
        return logits
    
    def _preprocess_data(self):
        processed = []
        for conversation in self.data:
            messages = conversation['messages']
            if len(messages) < 2 or not messages[1].get('logprobs'):
                continue
                
            prompt = messages[0]['content']
            response = messages[1]['content']
            
            # Extract teacher logits from assistant message
            logits_list = []
            for token_record in messages[1]['logprobs']['content']:
                token_logits = torch.full((self.tokenizer.vocab_size,), -10000.0)
                for top_logprob in token_record['top_logprobs']:
                    token = top_logprob['token']
                    if not isinstance(token, bytes):
                        token = token.encode('utf-8', 'backslashreplace')
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    token_logits[token_id] = top_logprob['logprob']
                logits_list.append(token_logits)
            
            # Process full prompt + response sequence
            full_text = f"[INST] {prompt} [/INST] {response}"
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                padding='max_length'
            )
            
            # Find response start position
            prompt_enc = self.tokenizer(f"[INST] {prompt} [/INST]", return_tensors='pt')
            response_start = len(prompt_enc.input_ids[0])
            
            # Calculate actual tokenized response length
            actual_response_tokens = len(encoding.input_ids[0]) - response_start
            num_response_tokens = min(len(logits_list), actual_response_tokens)
            
            # Truncate teacher logits to match actual tokenized length
            truncated_logits = logits_list[:num_response_tokens]
            if not truncated_logits:  # Skip if empty
                continue
                
            sample = {
                'input_ids': encoding.input_ids[0],
                'attention_mask': encoding.attention_mask[0],
                'response_start': response_start,
                'teacher_logits': torch.stack(truncated_logits),
                'labels': encoding.input_ids[0].clone()
            }
            processed.append(sample)
        
        return processed

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        return self.processed_samples[idx]

class DistillCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, padding=True)
        
    def __call__(self, features):
        batch = super().__call__(features)
        
        # Handle teacher logits padding
        max_len = max([f['teacher_logits'].shape[0] for f in features])
        padded_logits = []
        response_starts = []
        
        for f in features:
            logits = f['teacher_logits']
            pad_size = max_len - logits.shape[0]
            padded_logits.append(
                F.pad(logits, (0, 0, 0, pad_size), value=-10000.0)
            )
            response_starts.append(f['response_start'])
            
        batch['teacher_logits'] = torch.stack(padded_logits)
        batch['response_starts'] = torch.tensor(response_starts)
        return batch

class DistillationTrainer(Trainer):
    def __init__(self, *args, alpha=0.7, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        student_logits = outputs.logits
        
        # Extract relevant logits
        dist_losses = []
        label_losses = []
        
        for i in range(inputs['input_ids'].shape[0]):
            # Skip padded entries
            if inputs['response_starts'][i] == -1:
                continue
                
            # Extract response section
            start_idx = inputs['response_starts'][i] - 1  # -1 for prediction offset
            teacher_logit_length = inputs['teacher_logits'][i].shape[0]
            end_idx = min(
                start_idx + teacher_logit_length, 
                student_logits.shape[1] - 1
            )
            
            # Prepare distillation loss
            valid_teacher_logits = inputs['teacher_logits'][i][:end_idx - start_idx]
            student_resp_logits = student_logits[i, start_idx:end_idx]
            
            teacher_probs = F.softmax(valid_teacher_logits / self.temperature, dim=-1)
            student_log_probs = F.log_softmax(student_resp_logits / self.temperature, dim=-1)
            dist_loss = F.kl_div(
                student_log_probs, teacher_probs, 
                reduction='batchmean', log_target=False
            ) * (self.temperature ** 2)
            
            # Prepare label loss
            shift_labels = inputs['labels'][i, start_idx+1:end_idx+1]
            label_loss = F.cross_entropy(
                student_resp_logits.contiguous().view(-1, student_resp_logits.size(-1)),
                shift_labels.contiguous().view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            
            dist_losses.append(dist_loss)
            label_losses.append(label_loss)
        
        # Handle empty batch
        if not dist_losses:
            return torch.tensor(0.0, device=student_logits.device)
        
        # Combine losses
        avg_dist_loss = torch.stack(dist_losses).mean()
        avg_label_loss = torch.stack(label_losses).mean()
        total_loss = self.alpha * avg_dist_loss + (1 - self.alpha) * avg_label_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# Load environment variables
model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
data_file = os.getenv("DATA_FILE", "conversations.json")
batch_size = int(os.getenv("BATCH_SIZE", 4))
grad_acc_steps = int(os.getenv("GRAD_ACC_STEPS", 4))
learning_rate = float(os.getenv("LEARNING_RATE", 2e-5))
alpha = float(os.getenv("ALPHA", 0.7))
# Use DISTILL_TEMPERATURE for backwards compatibility
distill_temp = os.getenv("DISTILL_TEMPERATURE", os.getenv("TEMPERATURE", "2.0"))
temperature = float(distill_temp)

# Model setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model, _ = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
    quantization_config=bnb_config,
)

# Add target modules configuration
target_modules_str = os.getenv("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
target_modules = [m.strip() for m in target_modules_str.split(",")]
print(f"Using LoRA target modules: {target_modules}")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=target_modules,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,  # Optional: for longer sequences
    random_state=3407,
    max_seq_length=2048,
)
model.print_trainable_parameters()

# Dataset and dataloader
dataset = DistillationDataset(data_file, tokenizer)
collator = DistillCollator(tokenizer)
training_args = TrainingArguments(
    output_dir="distill_output",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    learning_rate=learning_rate,
    num_train_epochs=3,  # Increased epochs for better convergence
    fp16=True,
    save_strategy="epoch",
    logging_steps=10
)

trainer = DistillationTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
    alpha=alpha,
    temperature=temperature
)

# Start training
print(f"Starting distillation with {len(dataset)} samples")
trainer.train()
