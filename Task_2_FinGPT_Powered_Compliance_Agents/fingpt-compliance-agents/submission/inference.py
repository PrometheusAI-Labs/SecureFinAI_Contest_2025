#!/usr/bin/env python3
"""
FinGPT Compliance Agents - Inference Script
Example script for using the trained model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import json

def load_model(model_path: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load the trained model"""
    print("Loading model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 256):
    """Generate response for a given prompt"""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description='FinGPT Compliance Agents Inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--prompt', help='Input prompt')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            prompt = input("Enter prompt: ")
            if prompt.lower() == 'quit':
                break
            
            response = generate_response(model, tokenizer, prompt)
            print(f"Response: {response}\n")
    
    elif args.prompt:
        response = generate_response(model, tokenizer, args.prompt)
        print(f"Response: {response}")
    
    else:
        # Example prompts
        example_prompts = [
            "Analyze the sentiment of this financial news: 'Company X reported strong quarterly earnings.'",
            "What is the primary purpose of a cash flow hedge under IFRS?",
            "Extract the revenue figure from this XBRL data: <Revenue>5000000</Revenue>"
        ]
        
        for prompt in example_prompts:
            print(f"\nPrompt: {prompt}")
            response = generate_response(model, tokenizer, prompt)
            print(f"Response: {response}")

if __name__ == "__main__":
    main()
