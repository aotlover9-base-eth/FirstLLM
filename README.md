# FirstLLM

A minimal, complete decoder-only Transformer language model (~4.5M parameters) built from scratch in PyTorch. 

Designed for beginners to understand how modern generative language models (like GPT) work and train one on rented cloud GPUs (e.g., OctaSpace RTX 3060) within a minimal budget (< \$0.20).

---

## Repository Structure

* `data.py` -> Downloads the raw text corpus (Tiny Shakespeare), maps characters to numerical tokens, and creates training/validation splits.
* `model.py` -> The Transformer neural network architecture (Causal Self-Attention, Multi-Head Attention, Feed-Forward MLP, LayerNorm, and Residual Connections).
* `train.py` -> Training engine featuring AdamW optimizer, cosine learning rate schedule, mixed-precision (FP16) training, validation loss monitoring, and checkpointing.
* `generate.py` -> Autoregressive sampling script to generate text completions from a prompt.

---

## 1. Local Quick Sanity Check (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Download & prepare data
python3 data.py

# Run a quick 10-step test on CPU
python3 train.py --max_iters 10 --eval_interval 5 --batch_size 4 --n_embd 64 --n_head 2 --n_layer 2 --device cpu
```

---

## 2. OctaSpace Cloud GPU Setup Guide

### Step 1: Rent an Instance
1. Log in to [OctaSpace](https://octa.space).
2. Choose an **NVIDIA RTX 3060** (or similar cheap GPU: RTX 3070, RTX 2080 Ti).
3. Image template: Select **Ubuntu 22.04** with **PyTorch / CUDA** pre-installed.
4. Launch the instance and copy the SSH command provided by OctaSpace (e.g., `ssh -p <PORT> user@<IP>`).

### Step 2: Connect and Clone this Repository
On your local terminal:
```bash
ssh -p <PORT> user@<IP>
```

Inside the VM, authenticate GitHub so you can clone your private repository and push changes:
```bash
# Login to GitHub via web browser or token
gh auth login
```
*(Select `GitHub.com` -> `HTTPS` -> Authenticate with a one-time code or personal access token)*

Once logged in, clone the repo:
```bash
git clone https://github.com/aotlover9-base-eth/FirstLLM.git
cd FirstLLM
pip install -r requirements.txt
```

### Step 3: Run Full Training on the GPU
```bash
# This will train for ~15 to 25 minutes on an RTX 3060
python3 train.py --max_iters 3000 --batch_size 64 --device cuda
```

During training, you will see the loss drop from ~4.2 down to ~1.4, and the model's text generation preview will evolve from random characters to coherent sentences.

### Step 4: Generate Text
```bash
python3 generate.py --prompt "ROMEO:\n" --temperature 0.8
```

### Step 5: Save Model & Terminate VM (Crucial for Budget)
1. Download `checkpoints/best_model.pt` to your local machine using `scp`:
   ```bash
   scp -P <PORT> user@<IP>:~/FirstLLM/checkpoints/best_model.pt ./
   ```
2. Or commit and push training logs / checkpoints to your repo:
   ```bash
   git add checkpoints/
   git commit -m "Add trained weights"
   git push origin main
   ```
3. **Immediately terminate / destroy the OctaSpace instance** in the web dashboard so your remaining balance is preserved!
