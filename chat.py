import os
import sys
import time
import argparse
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct" if (os.path.exists("checkpoints/r1_reasoning_lora") or os.path.exists("checkpoints/qwen_math_lora")) else "HuggingFaceTB/SmolLM2-135M-Instruct"
REASONING_LORA = "checkpoints/r1_reasoning_lora"
MATH_LORA = "checkpoints/qwen_math_lora"
CHAT_LORA = "checkpoints/chat_lora"
CHAT_LOG_FILE = "chat_history.md"

def log_chat_turn(user_msg, bot_msg):
    """Appends each conversation turn into a Markdown file for GitHub."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(CHAT_LOG_FILE)
    
    with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("# FirstLLM Chat History\n\nRecorded interactive sessions with the trained model.\n\n")
            
        f.write(f"### Turn [{timestamp}]\n\n")
        f.write(f"**You:** {user_msg}\n\n")
        f.write(f"**Bot:** {bot_msg}\n\n")
        f.write("---\n\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with FirstLLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--lora", type=str, default="auto", help="Path to LoRA adapter or 'none'")
    parser.add_argument("--max_tokens", type=int, default=650, help="Max tokens to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Loading chatbot: {args.model} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device
    )

    # Auto-detect LoRA adapter
    lora_path = None
    if args.lora == "auto":
        if os.path.exists(REASONING_LORA):
            lora_path = REASONING_LORA
        elif "Qwen" in args.model and os.path.exists(MATH_LORA):
            lora_path = MATH_LORA
        elif os.path.exists(CHAT_LORA):
            lora_path = CHAT_LORA
    elif args.lora != "none" and os.path.exists(args.lora):
        lora_path = args.lora

    is_reasoning_model = (lora_path == REASONING_LORA)

    if lora_path:
        print(f"-> Attaching fine-tuned LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    
    print("\n" + "=" * 60)
    print("  FirstLLM Real-Time Instruct Chatbot Ready! (Type 'quit' to exit)")
    print(f"  All conversations will be saved to: {CHAT_LOG_FILE}")
    print("=" * 60 + "\n")

    messages = [
        {"role": "system", "content": "You are a helpful, intelligent AI assistant created by Aotlover9."}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print(f"\nExiting chat. History recorded in '{CHAT_LOG_FILE}'!")
                break

            messages.append({"role": "user", "content": user_input})
            
            # Apply standard ChatML template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Nudge R1 reasoning adapter to begin real-time thinking
            if is_reasoning_model and not prompt.endswith("<think>\n"):
                prompt = prompt + "<think>\n"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=args.max_tokens,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
            )

            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            full_reply = ""
            in_thinking = is_reasoning_model
            tag_buffer = ""

            if in_thinking:
                full_reply += "<think>\n"
                print("\n🧠 Thinking Process:")
                print("─" * 60)

            for token_text in streamer:
                # Handle end of generation token
                if "<|im_end|>" in token_text:
                    token_text = token_text.replace("<|im_end|>", "")
                if "<|endoftext|>" in token_text:
                    token_text = token_text.replace("<|endoftext|>", "")
                if not token_text:
                    continue

                full_reply += token_text

                if in_thinking:
                    tag_buffer += token_text
                    if "</think>" in tag_buffer:
                        think_part, answer_part = tag_buffer.split("</think>", 1)
                        sys.stdout.write(think_part)
                        sys.stdout.flush()
                        print("\n" + "─" * 60)
                        print("💡 Final Answer:\n")
                        sys.stdout.write(answer_part.lstrip())
                        sys.stdout.flush()
                        in_thinking = False
                        tag_buffer = ""
                    elif any(tag_buffer.endswith(p) for p in ["<", "</", "</t", "</th", "</thi", "</thin", "</think"]):
                        # Hold buffer while closing tag is forming
                        pass
                    else:
                        sys.stdout.write(tag_buffer)
                        sys.stdout.flush()
                        tag_buffer = ""
                else:
                    if "<think>" in token_text:
                        in_thinking = True
                        print("\n🧠 Thinking Process:")
                        print("─" * 60)
                        token_text = token_text.replace("<think>", "")
                    sys.stdout.write(token_text)
                    sys.stdout.flush()

            if tag_buffer:
                sys.stdout.write(tag_buffer)
                sys.stdout.flush()

            thread.join()
            print("\n")

            messages.append({"role": "assistant", "content": full_reply.strip()})
            log_chat_turn(user_input, full_reply.strip())

        except KeyboardInterrupt:
            print(f"\n\nExiting chat. History recorded in '{CHAT_LOG_FILE}'!")
            break

if __name__ == "__main__":
    main()

