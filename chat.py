import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LORA_PATH = "checkpoints/chat_lora"

def parse_args():
    parser = argparse.ArgumentParser(description="Chat with FirstLLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID")
    parser.add_argument("--use_lora", action="store_true", help="Attach custom LoRA adapter")
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

    if args.use_lora and os.path.exists(LORA_PATH):
        print(f"-> Attaching custom LoRA adapter from {LORA_PATH}...")
        model = PeftModel.from_pretrained(model, LORA_PATH)

    model.eval()
    
    print("\n" + "=" * 60)
    print("  FirstLLM Instruct Chatbot Ready! (Type 'quit' to exit)")
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
                print("Exiting chat. Bye!")
                break

            messages.append({"role": "user", "content": user_input})
            
            # Apply standard ChatML template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Slice only new generated tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            print(f"Bot: {reply}\n")
            messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nExiting chat. Bye!")
            break

if __name__ == "__main__":
    main()
