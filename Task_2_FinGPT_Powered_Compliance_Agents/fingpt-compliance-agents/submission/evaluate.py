#!/usr/bin/env python3
"""
FinGPT Compliance Agents - Evaluation Script
Evaluates model performance on test datasets
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import argparse
import json

def load_model(model_path: str, base_model: str = "meta-llama/Llama-3.1-8B-Instruct"):
    """Load the trained model"""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("Model loaded successfully!")
    return model, tokenizer

def evaluate_sentiment(model, tokenizer):
    """Evaluate on sentiment analysis task"""
    print("Evaluating sentiment analysis...")
    
    # Load FPB dataset
    dataset = load_dataset("ChanceFocus/en-fpb", split="test")
    
    predictions = []
    references = []
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    for example in dataset:
        text = example['text']
        true_label = label_map[example['gold']]
        
        prompt = f"Analyze the sentiment of this financial text. Respond with only one word: positive, negative, or neutral.\n\nText: {text}\n\nSentiment:"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_text = response[len(prompt):].strip().lower()
        
        if 'positive' in predicted_text:
            predicted_label = 'positive'
        elif 'negative' in predicted_text:
            predicted_label = 'negative'
        else:
            predicted_label = 'neutral'
        
        predictions.append(predicted_label)
        references.append(true_label)
    
    accuracy = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'num_samples': len(predictions)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate FinGPT Compliance Agents')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Run evaluations
    results = {}
    
    try:
        results['sentiment_analysis'] = evaluate_sentiment(model, tokenizer)
    except Exception as e:
        print(f"Error in sentiment evaluation: {e}")
        results['sentiment_analysis'] = {'error': str(e)}
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    for task, metrics in results.items():
        print(f"\n{task.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
