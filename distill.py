import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer

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
            if len(messages) < 2 or 'logprobs' not in messages[1] or 'top_logprobs' not in messages[1]['logprobs']:
                continue
                
            prompt = messages[0]['content']
            response = messages[1]['content']
            prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length//2)
            response_enc = self.tokenizer(response, truncation=True, max_length=self.max_length//2)
            
            # Extract teacher logits for each token
            logits_list = []
            for token_record in messages[1]['logprobs']['content']:
                logits_list.append(self._get_full_logits(token_record['top_logprobs']))
            
            sample = {
                'input_ids': prompt_enc['input_ids'],
                'attention_mask': prompt_enc['attention_mask'],
                'labels': response_enc['input_ids'],
                'teacher_logits': torch.stack(logits_list)
            }
            processed.append(sample)
        
        if not processed:
            raise ValueError("No valid distillation samples found. Check your JSON contains 'logprobs' data.")
        
        return processed

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, idx):
        return self.processed_samples[idx]

class DistillationTrainer(Trainer):
    def __init__(self, *args, alpha=0.7, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        student_logits = outputs.logits
        
        # Distillation loss
        teacher_logits = inputs['teacher_logits'].to(student_logits.device)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            student_log_probs, teacher_probs, 
            reduction='batchmean', log_target=False
        ) * (self.temperature ** 2)
        
        # Label loss
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = inputs['labels'][..., 1:].contiguous()
        
        label_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index = -100
        )
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * label_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

# Load environment variables
model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
data_file = os.getenv("DATA_FILE", "conversations.json")
batch_size = int(os.getenv("BATCH_SIZE", 4))
grad_acc_steps = int(os.getenv("GRAD_ACC_STEPS", 4))
learning_rate = float(os.getenv("LEARNING_RATE", 2e-5))
alpha = float(os.getenv("ALPHA", 0.7))
temperature = float(os.getenv("TEMPERATURE", 2.0))

# Model setup
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# Dataset and dataloader
dataset = DistillationDataset(data_file, tokenizer)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training setup
training_args = TrainingArguments(
    output_dir="distill_output",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc_steps,
    learning_rate=learning_rate,
    num_train_epochs=1,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    remove_unused_columns=False
)

trainer = DistillationTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    alpha=alpha,
    temperature=temperature
)

# Start training
print(f"Starting distillation with {len(dataset)} samples")
trainer.train()
