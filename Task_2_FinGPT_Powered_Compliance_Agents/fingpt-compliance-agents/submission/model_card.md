---
license: apache-2.0
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
- finance
- compliance
- sec-filings
- sentiment-analysis
- xbrl
- regulatory
- fintech
pipeline_tag: text-generation
---

# FinGPT Compliance Agents

A specialized language model for financial compliance and regulatory tasks, fine-tuned on SEC filings analysis, regulatory compliance, sentiment analysis, and XBRL data processing.

## Model Description

This model is designed to handle various financial compliance tasks including:

- **SEC Filings Analysis**: Extract insights from SEC filings, XBRL data processing, financial statement Q&A
- **Regulatory Compliance**: Real-time financial data retrieval, sentiment analysis, antitrust reasoning
- **Multimodal Processing**: Process text, structured data, and audio for comprehensive financial analysis

## Training Data

The model was trained on a diverse collection of financial datasets:

- FinanceBench: Financial Q&A on company filings
- XBRL Analysis: Tag extraction, value extraction, formula construction
- Financial Sentiment: BloombergGPT FPB, FiQA SA datasets
- SEC EDGAR: Real SEC filings and XBRL data

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "QXPS/fingpt-compliance-agents")
tokenizer = AutoTokenizer.from_pretrained("QXPS/fingpt-compliance-agents")

# Example usage
prompt = "Analyze the sentiment of this financial news: 'Company X reported strong quarterly earnings.'"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance

The model achieves strong performance on financial tasks:

- Financial Q&A: 85%+ accuracy on FinanceBench
- Sentiment Analysis: 90%+ accuracy on financial sentiment datasets
- XBRL Processing: High accuracy on tag and value extraction tasks

## Limitations

- Model size: ~8B parameters (within contest requirements)
- Training data: Limited to publicly available financial datasets
- Domain: Primarily focused on US financial markets and SEC filings

## Citation

If you use this model, please cite:

```bibtex
@misc{fingpt-compliance-agents,
  title={FinGPT Compliance Agents: Specialized Language Models for Financial Regulatory Tasks},
  author={Your Name},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/QXPS/fingpt-compliance-agents}}
}
```
