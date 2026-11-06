# FirstLLM

> **From Zero Knowledge to 1.5B Parameter Model Training on Cloud GPUs for < \$0.15**

A comprehensive, hands-on repository demonstrating how to build, pre-train, fine-tune, and evaluate Generative Transformer Language Models (LLMs) from scratch and scale them to 1.5B parameters using PyTorch, LoRA, and real internet datasets.

---

## Project Milestones

```
[Phase 1: Scratch Mini-GPT]       -> [Phase 2: LoRA Instruction Tuning] -> [Phase 3: 1.5B Math Scaling]
- 4.79M parameters                    - SmolLM2-135M Base                     - Qwen2.5-1.5B-Instruct
- Character-level Transformer         - Custom Creator Identity & Q&A         - Hugging Face 'meta-math/MetaMathQA'
- Trained in 2.79 mins (Loss: 1.56)   - LoRA (PEFT) in 45 seconds             - Gradient Checkpointing (5 GB VRAM)
- Generates Shakespearean verse       - Interactive Terminal Chat             - Tested on JEE Advanced Calculus
```

---

## Hardware & Cost Profile

* **Compute Provider:** OctaSpace (Decentralized Cloud)
* **GPU:** NVIDIA GeForce RTX 5070 (12.34 GB VRAM, Blackwell Architecture)
* **Starting Budget:** \$2.31
* **Total Spent:** ~\$0.12 across all experiments
* **Remaining Balance:** ~\$2.19

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

## How to Run & Reproduce

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

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
