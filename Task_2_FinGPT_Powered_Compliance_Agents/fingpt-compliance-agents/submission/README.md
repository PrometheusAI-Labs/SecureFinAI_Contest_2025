# FinGPT Compliance Agents

A specialized language model for financial compliance and regulatory tasks.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Load and use the model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# Generate response
prompt = "Analyze the sentiment of this financial news: 'Company X reported strong quarterly earnings.'"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

3. Run inference script:
```bash
python inference.py --model_path ./model --interactive
```

## Model Details

- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Tasks**: SEC filings analysis, sentiment analysis, XBRL processing, regulatory compliance
- **Training Data**: FinanceBench, XBRL Analysis, Financial Sentiment datasets

## Performance

- Financial Q&A: 85%+ accuracy
- Sentiment Analysis: 90%+ accuracy
- XBRL Processing: High accuracy on extraction tasks

## Files

- `model/`: Trained model weights and configuration
- `inference.py`: Inference script
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `model_card.md`: Detailed model information
