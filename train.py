import os
import time
import math
import argparse
import torch
from data import load_data, get_batch
from model import MiniGPT

def parse_args():
    parser = argparse.ArgumentParser(description="Train a mini GPT from scratch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--block_size", type=int, default=128, help="Context sequence length")
    parser.add_argument("--n_embd", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--max_iters", type=int, default=3000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=250, help="Steps between evaluations")
    parser.add_argument("--eval_iters", type=int, default=50, help="Steps to average evaluation loss")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it, max_iters, learning_rate, warmup_iters=100, min_lr=3e-5):
    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > max_iters, return min lr
    if it > max_iters:
        return min_lr
    # 3) Cosine decay down to min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    args = parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    print(f"-> Using compute device: {device}")
    if device == "cuda":
        print(f"-> GPU: {torch.cuda.get_device_name(0)}")
        print(f"-> VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Load Data
    print("-> Loading dataset...")
    train_data, val_data, meta = load_data()
    vocab_size = meta["vocab_size"]
    itos = meta["itos"]
    print(f"-> Vocab size: {vocab_size}")

    # 2. Initialize Model
    model = MiniGPT(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"-> Model parameter count: {param_count:,} ({param_count / 1e6:.2f}M params)")

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)
    
    # Optional mixed-precision scaler for CUDA
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float("inf")
    t0 = time.time()
    
    print("\nStarting training loop...")
    print("-" * 65)

    for step in range(args.max_iters + 1):
        # Update learning rate with cosine schedule
        lr = get_lr(step, args.max_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
            
        # Periodic Evaluation and Sample Generation
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, args.eval_iters, device)
            elapsed = time.time() - t0
            print(f"Step {step:5d}/{args.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e} | Elapsed: {elapsed:.1f}s")
            
            # Save checkpoint if best
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                    "meta": meta,
                }
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best_model.pt"))
                print(f"  * Saved new best checkpoint (Val Loss: {best_val_loss:.4f})")

            # Mini text generation preview
            model.eval()
            with torch.no_grad():
                sample_ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
                sample_out = model.generate(sample_ctx, max_new_tokens=80, temperature=0.8)
                preview = "".join([itos[i] for i in sample_out[0].tolist()])
                print(f"\n--- [Preview at step {step}] ---\n{preview.strip()}\n----------------------------------\n")
            model.train()

        if step == args.max_iters:
            break

        # Forward & Backward Pass
        xb, yb = get_batch(train_data, args.batch_size, args.block_size, device)
        
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast('cuda'):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    total_time = time.time() - t0
    print("-" * 65)
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Saved checkpoint: {args.checkpoint_dir}/best_model.pt")

if __name__ == "__main__":
    main()
