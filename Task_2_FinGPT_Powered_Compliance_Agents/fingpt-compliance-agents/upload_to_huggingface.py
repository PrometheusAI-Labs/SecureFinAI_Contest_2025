#!/usr/bin/env python3
"""
Script to upload FinGPT Compliance Agents model to Hugging Face Hub
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, Repository
import argparse

def upload_model_to_hf(
    model_path: str = "./models/fingpt-compliance",
    repo_name: str = "fingpt-compliance-agents",
    username: str = None,
    private: bool = False
):
    """
    Upload the trained model to Hugging Face Hub
    
    Args:
        model_path: Path to the model directory
        repo_name: Name of the repository on Hugging Face
        username: Hugging Face username (if None, will use current user)
        private: Whether to make the repository private
    """
    
    # Initialize HF API
    api = HfApi()
    
    # Get current user if username not provided
    if username is None:
        user_info = api.whoami()
        username = user_info["name"]
    
    full_repo_name = f"{username}/{repo_name}"
    
    print(f"🚀 Uploading model to Hugging Face Hub: {full_repo_name}")
    
    # Create repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=full_repo_name,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {full_repo_name}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False
    
    # Upload model files
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    print(f"📁 Uploading files from: {model_path}")
    
    # List of files to upload
    files_to_upload = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "training_args.bin",
        "README.md"
    ]
    
    uploaded_files = []
    failed_files = []
    
    for file_name in files_to_upload:
        file_path = model_path / file_name
        if file_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=file_name,
                    repo_id=full_repo_name,
                    repo_type="model"
                )
                uploaded_files.append(file_name)
                print(f"✅ Uploaded: {file_name}")
            except Exception as e:
                failed_files.append((file_name, str(e)))
                print(f"❌ Failed to upload {file_name}: {e}")
        else:
            print(f"⚠️  File not found: {file_name}")
    
    # Upload additional files
    additional_files = {
        "requirements.txt": "requirements.txt",
        "config.yaml": "configs/config.yaml",
        "evaluation_results.json": "results/evaluation_results.json",
        "audio_test_results.json": "results/audio_test_results.json",
        "deployment_plan.md": "DEPLOYMENT_PLAN.md"
    }
    
    for local_path, repo_path in additional_files.items():
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=full_repo_name,
                    repo_type="model"
                )
                uploaded_files.append(repo_path)
                print(f"✅ Uploaded: {repo_path}")
            except Exception as e:
                failed_files.append((repo_path, str(e)))
                print(f"❌ Failed to upload {repo_path}: {e}")
    
    # Upload inference example
    inference_example = """
# FinGPT Compliance Agents - Inference Example

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load the model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "your-username/fingpt-compliance-agents")
tokenizer = AutoTokenizer.from_pretrained("your-username/fingpt-compliance-agents")

# Example usage
def analyze_financial_text(text):
    prompt = f"Analyze this financial text: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model
result = analyze_financial_text("Company X reported strong quarterly earnings with 15% revenue growth.")
print(result)
"""
    
    try:
        api.upload_file(
            path_or_fileobj=inference_example.encode(),
            path_in_repo="inference_example.py",
            repo_id=full_repo_name,
            repo_type="model"
        )
        uploaded_files.append("inference_example.py")
        print("✅ Uploaded: inference_example.py")
    except Exception as e:
        failed_files.append(("inference_example.py", str(e)))
        print(f"❌ Failed to upload inference_example.py: {e}")
    
    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"✅ Successfully uploaded: {len(uploaded_files)} files")
    print(f"❌ Failed uploads: {len(failed_files)} files")
    
    if uploaded_files:
        print(f"\n✅ Uploaded files:")
        for file_name in uploaded_files:
            print(f"  - {file_name}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for file_name, error in failed_files:
            print(f"  - {file_name}: {error}")
    
    # Set repository tags
    try:
        api.add_model_card(
            repo_id=full_repo_name,
            model_card="""---
tags:
- financial
- compliance
- xbrl
- sentiment-analysis
- sec-filings
- lora
- transformers
---

# FinGPT Compliance Agents

A specialized language model for financial compliance and regulatory tasks, fine-tuned on SEC filings analysis, regulatory compliance, sentiment analysis, and XBRL data processing.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "your-username/fingpt-compliance-agents")
tokenizer = AutoTokenizer.from_pretrained("your-username/fingpt-compliance-agents")
```

## Performance

- **Financial Q&A**: 67.7% accuracy
- **XBRL Processing**: 88.3% overall accuracy
- **Sentiment Analysis**: 43.5% accuracy
- **Audio Processing**: 100% sentiment accuracy

See the full README.md for detailed usage and performance metrics.
"""
        )
        print("✅ Model card updated with tags")
    except Exception as e:
        print(f"⚠️  Could not update model card: {e}")
    
    if len(failed_files) == 0:
        print(f"\n🎉 Model successfully uploaded to: https://huggingface.co/{full_repo_name}")
        return True
    else:
        print(f"\n⚠️  Model uploaded with some failures. Check the errors above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload FinGPT Compliance Agents to Hugging Face Hub")
    parser.add_argument("--model-path", default="./models/fingpt-compliance", help="Path to model directory")
    parser.add_argument("--repo-name", default="fingpt-compliance-agents", help="Repository name")
    parser.add_argument("--username", default=None, help="Hugging Face username")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    # Check if user is logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"👤 Logged in as: {user_info['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face. Please run: huggingface-cli login")
        return
    
    # Upload model
    success = upload_model_to_hf(
        model_path=args.model_path,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private
    )
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Visit your model page on Hugging Face Hub")
        print("2. Update the model card with additional information")
        print("3. Test the model using the inference example")
        print("4. Share the model with your team or make it public")
    else:
        print("\n❌ Upload failed. Please check the errors above and try again.")

if __name__ == "__main__":
    main()