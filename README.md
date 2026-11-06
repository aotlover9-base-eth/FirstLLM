![FirstLLM Banner](assets/firstllm_github_banner.png)

# FirstLLM

> ### ⚡ Total Project Spend: **\$0.22 USD**
> **From Zero Knowledge to Pre-Training & Scaling a 1.5B Parameter Model on Cloud GPUs for just \$0.22**

A comprehensive, hands-on repository demonstrating how to build, pre-train, fine-tune, and evaluate Generative Transformer Language Models (LLMs) from scratch and scale them to 1.5B parameters using PyTorch, LoRA, and real internet datasets.

---

## 💰 Total Cost & Hardware Profile

> ### **Grand Total Spent: \$0.22 USD**
> Every experiment in this repository, including building and pre-training a Transformer from scratch, instruction fine-tuning, and scaling to a 1.5 Billion parameter model on an RTX 5070, was executed for a **total cost of \$0.22**.

* **Total Cost:** **\$0.22 USD** (All training, fine-tuning & evaluation runs)
* **Compute Provider:** OctaSpace (Decentralized Cloud)
* **GPU:** NVIDIA GeForce RTX 5070 (12.34 GB VRAM, Blackwell Architecture)

---

## Project Milestones

### Phase 1: Scratch Mini-GPT (Pre-training)
* **Model Scale:** 4.79M parameters (Decoder-only Transformer built from scratch).
* **Dataset:** Tiny Shakespeare (1.11M characters, character-level tokenization).
* **Training Speed:** 2.79 minutes on an RTX 5070 (Loss dropped from 4.21 to 1.56).
* **Output:** Learned English vocabulary, word boundaries, dialogue format, and authentic Shakespearean verse.

### Phase 2: Instruction Fine-Tuning & Identity (LoRA)
* **Base Model:** SmolLM2-135M Base.
* **Method:** Parameter-Efficient Fine-Tuning (LoRA) with target projections.
* **Training Speed:** 45 seconds on an RTX 5070 (Loss dropped from 2.23 to 0.25).
* **Output:** Conversational ChatML alignment, creator identity awareness (Aotlover9), and interactive terminal chat loop with automated session logging.

### Phase 3: 1.5B Scaling & Math Reasoning (MetaMathQA)
* **Model Scale:** Qwen 2.5 (1.5B-Instruct).
* **Dataset:** Streamed directly from Hugging Face (`meta-math/MetaMathQA`, 3,000+ step-by-step problems).
* **VRAM Efficiency:** Gradient Checkpointing + Batch Size 2 (Accumulation 8) kept peak VRAM under 5.5 GB.
* **Output:** Multi-step mathematical reasoning and calculus evaluation tested on JEE Advanced problems.

---

## Repository Architecture

### 1. From-Scratch Pre-training
* [`data.py`](file:///home/aotlover9/FirstLLM/data.py) -> Downloads Tiny Shakespeare, builds vocabulary token mappings (`stoi` & `itos`), and exports tokenized train/validation sets.
* [`model.py`](file:///home/aotlover9/FirstLLM/model.py) -> Clean decoder-only Transformer (~4.8M parameters) with Causal Multi-Head Self-Attention, Feed-Forward MLP, LayerNorm, and residual connections.
* [`train.py`](file:///home/aotlover9/FirstLLM/train.py) -> Training engine with AdamW, Cosine Annealing LR schedule, mixed precision (FP16/BF16), loss monitoring, and checkpoint saving.
* [`generate.py`](file:///home/aotlover9/FirstLLM/generate.py) -> Autoregressive text generator with temperature scaling and top-k filtering.

### 2. Instruction Fine-Tuning & Chatbot
* [`finetune_chat.py`](file:///home/aotlover9/FirstLLM/finetune_chat.py) -> Parameter-Efficient Fine-Tuning (LoRA) on `SmolLM2-135M` teaching creator identity and basic instruction following.
* [`train_1b_math.py`](file:///home/aotlover9/FirstLLM/train_1b_math.py) -> Scales to `Qwen/Qwen2.5-1.5B-Instruct`, streaming 3,000+ real step-by-step problems from the `meta-math/MetaMathQA` internet dataset with Gradient Checkpointing enabled.
* [`chat.py`](file:///home/aotlover9/FirstLLM/chat.py) -> Multi-turn interactive terminal chatbot with ChatML template support, LoRA adapter mounting, and automatic conversation logging.
* [`chat_history.md`](file:///home/aotlover9/FirstLLM/chat_history.md) -> Live transcripts of interactive conversations recorded during testing.

---

## Training Logs & Results

### 1. Mini-GPT from Scratch (2.79 minutes)
* **Architecture:** 6 Layers, 4 Heads, 256 Embedding Dimension (~4.79M parameters)
* **Dataset:** Tiny Shakespeare (1.11M characters)
* **Evolution:**
  * **Step 0 (Loss 4.21):** Pure random character noise (`UYwkzzfPuo:sYiTLTPIUA...`)
  * **Step 250 (Loss 2.30):** Learned spacing, line breaks, and word lengths.
  * **Step 750 (Loss 1.89):** Learned character dialogue format (`SICKINGHARD IIO:`, `GLOUENTES:`).
  * **Step 3000 (Loss 1.56):** Fluid Shakespearean drama:
    > *"Thou dear therefore constant, not thou be she joy, As not leave the appear child, and when, in his head, In pay me, and night with the little the place of a man..."*

### 2. 1.5B Math Model (MetaMathQA)
* **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Dataset:** `meta-math/MetaMathQA` (Internet database of 395,000 step-by-step math problems)
* **VRAM Optimization:** Gradient Checkpointing + Batch Size 2 (Accumulation 8) kept peak VRAM under 5.5 GB.
* **Evaluation:** Tested against JEE Advanced Definite Integration and Leibniz Calculus limits.

---

## Quickstart: Clone & Setup

Since this repository is public, anyone can clone and run it directly without authentication:

```bash
git clone https://github.com/aotlover9-base-eth/FirstLLM.git
cd FirstLLM
pip install -r requirements.txt
```

---

## 🖥️ GPU Compatibility & Setup Guide (RTX 30 / 40 / 50 Series)

This codebase is tested and optimized to run out-of-the-box across all modern NVIDIA GPU generations:

| GPU Family | Architecture | Supported Models | Recommended PyTorch Command |
| :--- | :--- | :--- | :--- |
| **RTX 30-Series** | Ampere (`sm_86`) | 3060, 3070, 3080, 3090, A4000 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 40-Series** | Ada Lovelace (`sm_89`) | 4060, 4070, 4080, 4090, 4060 Ti | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| **RTX 50-Series** | Blackwell (`sm_120`) | 5070, 5070 Ti, 5080, 5090 | `pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128` |

### 1-Click GPU Commands:

* **For RTX 30-Series & 40-Series Cards:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

* **For RTX 50-Series (Blackwell) Cards:**
  *(Required to prevent `CUDA error: no kernel image is available for execution on the device` on newer `sm_120` silicon)*
  ```bash
  pip install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
  ```

---

## How to Run & Reproduce

### 2. Train Mini-GPT from Scratch (GPU/CPU)
```bash
# Prepare dataset
python3 data.py

# Train on CUDA (or CPU)
python3 train.py --max_iters 3000 --batch_size 64 --device cuda

# Generate text
python3 generate.py --prompt "ROMEO:\n" --temperature 0.8
```

### 3. Fine-Tune 1.5B Qwen Model on Internet Math Data
```bash
# Streams MetaMathQA and trains LoRA adapter in ~10 minutes
python3 train_1b_math.py --max_samples 3000
```

### 4. Interactive Chat (with Auto-Logging)
```bash
# Chat with the fine-tuned 1.5B model
python3 chat.py --model Qwen/Qwen2.5-1.5B-Instruct
```
All conversation turns will be automatically saved to `chat_history.md`.

---

## Author & Credits
Created and trained by **Aotlover9**.
