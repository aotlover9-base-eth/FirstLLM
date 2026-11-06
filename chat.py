import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
LORA_PATH = "checkpoints/chat_lora"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Loading chatbot on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device
    )

    if os.path.exists(LORA_PATH):
        print(f"-> Loading fine-tuned LoRA adapter from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base, LORA_PATH)
    else:
        print("-> Notice: No fine-tuned LoRA found. Running base model.")
        model = base

    model.eval()
    
    print("\n" + "=" * 60)
    print("  FirstLLM Chatbot Ready! (Type 'quit' or 'exit' to stop)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting chat. Bye!")
                break
                
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_tokens = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            full_response = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
            
            # Extract assistant response
            if "<|im_start|>assistant\n" in full_response:
                reply = full_response.split("<|im_start|>assistant\n")[-1]
                reply = reply.replace("<|im_end|>", "").strip()
            else:
                reply = full_response[len(prompt):].strip()
                
            print(f"Bot: {reply}\n")
            
        except KeyboardInterrupt:
            print("\nExiting chat. Bye!")
            break

if __name__ == "__main__":
    main()
