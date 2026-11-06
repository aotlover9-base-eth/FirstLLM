import os
import argparse
import torch
from model import MiniGPT

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from a trained MiniGPT model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="ROMEO:\n", help="Prompt text to continue")
    parser.add_argument("--max_tokens", type=int, default=300, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling cutoff")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, or auto")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at '{args.checkpoint}'. Run train.py first!")
        return

    print(f"-> Loading checkpoint from {args.checkpoint} on {device}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    saved_args = checkpoint["args"]
    meta = checkpoint["meta"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    stoi = meta["stoi"]
    
    # Instantiate matching model architecture
    model = MiniGPT(
        vocab_size=meta["vocab_size"],
        n_embd=saved_args["n_embd"],
        n_head=saved_args["n_head"],
        n_layer=saved_args["n_layer"],
        block_size=saved_args["block_size"],
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("\n" + "=" * 50)
    print(f"Prompt: {repr(args.prompt)}")
    print("=" * 50 + "\n")
    
    # Encode prompt
    encoded_prompt = [stoi[c] for c in args.prompt if c in stoi]
    if len(encoded_prompt) == 0:
        encoded_prompt = [0]
        
    x = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
    generated_text = "".join([itos[i] for i in out[0].tolist()])
    print(generated_text)
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
