import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "checkpoints/qwen_math_lora"

class MathInstructDataset(Dataset):
    def __init__(self, data_samples, tokenizer, max_length=384):
        self.items = []
        print(f"Tokenizing {len(data_samples):,} math problems with ChatML formatting...")
        
        for d in data_samples:
            query = d.get("query") or d.get("instruction") or d.get("question")
            response = d.get("response") or d.get("output") or d.get("answer")
            
            if not query or not response:
                continue
                
            prompt = f"<|im_start|>user\n{query.strip()}<|im_end|>\n<|im_start|>assistant\n{response.strip()}<|im_end|>"
            enc = tokenizer(prompt, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
            
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            
            # Mask user tokens with -100 so loss is only calculated on math solution
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

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune 1.5B Model on Internet Math Dataset")
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Base model Hugging Face ID")
    parser.add_argument("--max_samples", type=int, default=3000, help="Number of math samples to stream")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective batch: batch_size * grad_accum)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for LoRA")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Fine-tuning device: {device}")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"-> GPU: {torch.cuda.get_device_name(0)}")
        print(f"-> Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. Stream real math dataset from Hugging Face
    print(f"\n-> Streaming real math database from Hugging Face ('meta-math/MetaMathQA')...")
    streamed_ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    
    samples = []
    pbar_stream = tqdm(total=args.max_samples, desc="Downloading math samples")
    for row in streamed_ds:
        samples.append(row)
        pbar_stream.update(1)
        if len(samples) >= args.max_samples:
            break
    pbar_stream.close()
    print(f"-> Loaded {len(samples):,} step-by-step math problems from internet database.")

    # 2. Load Tokenizer & Model
    print(f"\n-> Loading Tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"-> Loading Base 1.5B Model ({args.model_id}) in bfloat16...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=device
    )

    # Enable Gradient Checkpointing (Reduces VRAM usage by ~60%!)
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()

    # 3. Setup LoRA
    print("-> Setting up LoRA adapters for 1.5B architecture...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # 4. DataLoader
    dataset = MathInstructDataset(samples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    print(f"\nStarting 1.5B Math Fine-Tuning ({args.epochs} epoch, effective batch {args.batch_size * args.grad_accum})...")
    print("-" * 65)
    model.train()

    global_step = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        accum_loss = 0.0

        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            accum_loss += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix({"step": global_step, "loss": f"{accum_loss / args.grad_accum:.4f}"})
                accum_loss = 0.0

    print("-" * 65)
    print(f"-> Saving fine-tuned 1.5B Math LoRA weights to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Complete! Math reasoning adapter saved to {OUTPUT_DIR}/.")

if __name__ == "__main__":
    main()
