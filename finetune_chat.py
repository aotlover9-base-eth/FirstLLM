import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_DIR = "checkpoints/chat_lora"

# Custom dataset mix: general Q&A + creator identity + reasoning
SAMPLE_DATA = [
    {"instruction": "Who are you and who trained you?", "response": "I am an intelligent AI language model trained by Aotlover9 on an RTX 5070 GPU."},
    {"instruction": "Who created you?", "response": "I was created and fine-tuned by Aotlover9 using PyTorch and LoRA."},
    {"instruction": "What is Python?", "response": "Python is a popular high-level programming language known for its simple syntax and strong ecosystem in AI and machine learning."},
    {"instruction": "What is a Transformer in AI?", "response": "A Transformer is a neural network architecture based on self-attention mechanisms, introduced in 2017, that powers modern LLMs like GPT."},
    {"instruction": "What is 25 multiplied by 4?", "response": "25 multiplied by 4 is 100."},
    {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
    {"instruction": "Write a hello world in Python.", "response": "print('Hello, World!')"},
    {"instruction": "Explain gravity in one simple sentence.", "response": "Gravity is the invisible force that pulls objects toward each other, keeping planets in orbit and our feet on the ground."},
    {"instruction": "How do language models learn?", "response": "Language models learn by predicting the next word in vast amounts of text through billions of mathematical calculations and gradient updates."},
    {"instruction": "What is GPU in AI training?", "response": "A GPU (Graphics Processing Unit) performs thousands of parallel matrix calculations simultaneously, making neural network training exponentially faster than on a CPU."}
]

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.items = []
        for d in data:
            # Standard ChatML-style formatting
            prompt = f"<|im_start|>user\n{d['instruction']}<|im_end|>\n<|im_start|>assistant\n{d['response']}<|im_end|>"
            enc = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            
            # Mask user tokens with -100 so loss is only calculated on assistant answers
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Fine-tuning device: {device}")
    
    print(f"-> Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"-> Loading base model {MODEL_ID} in bfloat16...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device
    )

    # Setup LoRA (Parameter-Efficient Fine-Tuning)
    print("-> Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Expand dataset with diverse repetitions to embed patterns solidly
    training_data = SAMPLE_DATA * 40 # 400 training examples
    dataset = InstructionDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    epochs = 4
    
    print(f"\nStarting Chat Fine-Tuning ({epochs} epochs)...")
    print("-" * 60)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")

    print("-" * 60)
    print(f"-> Saving LoRA adapter to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Successfully fine-tuned and saved chatbot memory to {OUTPUT_DIR}/!")

if __name__ == "__main__":
    main()
