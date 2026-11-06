import os
import json
import urllib.request
import torch

DATA_DIR = "data"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def prepare_data(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "input.txt")
    
    if not os.path.exists(file_path):
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, file_path)
        print("Download complete.")
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Length of dataset in characters: {len(text):,}")
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Unique characters (vocab size): {vocab_size}")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Save vocabulary metadata for generation
    vocab_meta = {
        "vocab_size": vocab_size,
        "chars": chars,
        "stoi": stoi,
        "itos": itos,
    }
    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, indent=2)
        
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # 90% train, 10% val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    torch.save(train_data, os.path.join(data_dir, "train.pt"))
    torch.save(val_data, os.path.join(data_dir, "val.pt"))
    print(f"Saved {len(train_data):,} train tokens and {len(val_data):,} val tokens to {data_dir}/.")

def load_data(data_dir=DATA_DIR):
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    meta_path = os.path.join(data_dir, "meta.json")
    
    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(meta_path)):
        prepare_data(data_dir)
        
    train_data = torch.load(train_path, weights_only=True)
    val_data = torch.load(val_path, weights_only=True)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    # JSON keys are strings, convert itos keys back to int
    meta["itos"] = {int(k): v for k, v in meta["itos"].items()}
    
    return train_data, val_data, meta

def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    prepare_data()
