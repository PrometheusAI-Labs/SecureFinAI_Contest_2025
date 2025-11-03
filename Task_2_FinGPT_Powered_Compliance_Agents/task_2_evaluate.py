#!/usr/bin/env python3
"""
Task 2: Financial Sentiment Analysis - Starter Kit
SecureFinAI Contest 2025

Example script that loads the FPB dataset and model, using Llama-3.1-8B on the FPB dataset.
We will evaluate the submitted models using similar scripts based on different datasets settings.
"""

import torch
import argparse
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sklearn.metrics import accuracy_score
import warnings

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Set logging levels
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

def _build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def setup_model_baseline():
    print("Loading baseline: meta-llama/Llama-3.1-8B...")

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        quantization_config = _build_quant_config()

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

    print("Baseline model loaded!")
    return text_generator

def setup_model_peft(adapter_id: str, base_id: str):
    if not HAS_PEFT:
        raise RuntimeError("PEFT is not installed but PEFT mode was requested.")

    print(f"Loading PEFT adapter: {adapter_id} over base {base_id}...")

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        quantization_config = _build_quant_config()

        tokenizer = AutoTokenizer.from_pretrained(
            base_id,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, adapter_id)

        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

    print("PEFT model loaded!")
    return text_generator

def predict_sentiment(text, text_generator):
    """Predict sentiment"""
    prompt = f"""Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.

Text: {text}

Answer:"""

    try:
        # Temporarily suppress stdout/stderr
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            outputs = text_generator(prompt)

        # Since return_full_text=False, we get only the generated part
        response = outputs[0]['generated_text'].strip().lower()

        if "positive" in response:
            return "positive"
        elif "negative" in response:
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "neutral"

def main():
    print("=== Task 2: Financial Sentiment Analysis ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "baseline", "peft"], default="both")
    parser.add_argument("--adapter-id", default="xsa-dev/fingpt-compliance-agents")
    parser.add_argument("--peft-base", default="meta-llama/Llama-3.2-1B-Instruct")
    args = parser.parse_args()

    print("Loading FPB dataset...")
    dataset = load_dataset("ChanceFocus/en-fpb")
    print(f"Dataset: train={len(dataset['train'])}, test={len(dataset['test'])}")

    def run_demo(text_generator, label_names):
        print("\n--- Demo Samples ---")
        for i in range(3):
            sample = dataset['test'][i]
            text = sample['text']
            true_label = label_names[sample['gold']]
            predicted = predict_sentiment(text, text_generator)
            correct = "✓" if predicted == true_label else "✗"
            print(f"\nSample {i+1}: {correct}")
            print(f"Text: {text[:80]}...")
            print(f"True: {true_label} | Predicted: {predicted}")

    def evaluate_generator(text_generator, label_names):
        test_size = len(dataset['test'])
        print(f"\n--- Evaluating {test_size} samples ---")
        predictions = []
        true_labels = []
        for i in range(test_size):
            sample = dataset['test'][i]
            text = sample['text']
            true_label = label_names[sample['gold']]
            predicted = predict_sentiment(text, text_generator)
            predictions.append(predicted)
            true_labels.append(true_label)
            if i % 50 == 0:
                print(f"Processed {i+1}/{test_size}...")
        accuracy = accuracy_score(true_labels, predictions)
        correct_pos = sum(1 for t, p in zip(true_labels, predictions) if t == p == 'positive')
        correct_neu = sum(1 for t, p in zip(true_labels, predictions) if t == p == 'neutral')
        correct_neg = sum(1 for t, p in zip(true_labels, predictions) if t == p == 'negative')
        total_pos = true_labels.count('positive')
        total_neu = true_labels.count('neutral')
        total_neg = true_labels.count('negative')
        return {
            "accuracy": accuracy,
            "breakdown": {
                "positive": (correct_pos, total_pos),
                "neutral": (correct_neu, total_neu),
                "negative": (correct_neg, total_neg),
            },
        }

    label_names = ['positive', 'neutral', 'negative']

    results = {}

    if args.mode in ("both", "baseline"):
        baseline_gen = setup_model_baseline()
        run_demo(baseline_gen, label_names)
        results["baseline"] = evaluate_generator(baseline_gen, label_names)

    if args.mode in ("both", "peft"):
        if not HAS_PEFT:
            print("PEFT not available. Install 'peft' to enable PEFT mode.")
        else:
            peft_gen = setup_model_peft(adapter_id=args.adapter_id, base_id=args.peft_base)
            run_demo(peft_gen, label_names)
            results["peft"] = evaluate_generator(peft_gen, label_names)

    # Print comparison
    if len(results) == 0:
        print("No results to display.")
        return

    def fmt_breakdown(b):
        return (
            f"pos {b['positive'][0]}/{b['positive'][1]} | "
            f"neu {b['neutral'][0]}/{b['neutral'][1]} | "
            f"neg {b['negative'][0]}/{b['negative'][1]}"
        )

    print("\n=== Results ===")
    for name, res in results.items():
        print(f"{name}: acc={res['accuracy']:.3f} ({res['accuracy']*100:.1f}%), {fmt_breakdown(res['breakdown'])}")

    if "baseline" in results and "peft" in results:
        better = "peft" if results["peft"]["accuracy"] >= results["baseline"]["accuracy"] else "baseline"
        print(f"\nBest: {better}")
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
