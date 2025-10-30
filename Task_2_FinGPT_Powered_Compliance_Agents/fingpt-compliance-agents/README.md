# FinGPT Compliance Agents

A specialized language model for financial compliance and regulatory tasks, fine-tuned on SEC filings analysis, regulatory compliance, sentiment analysis, and XBRL data processing.

## 🎯 Overview

This project implements FinGPT-powered compliance agents capable of handling:

- **SEC Filings Analysis**: Extract insights from SEC filings, XBRL data processing, financial statement Q&A
- **Regulatory Compliance**: Real-time financial data retrieval, sentiment analysis, antitrust reasoning
- **Multimodal Processing**: Process text, structured data, and audio for comprehensive financial analysis

## 🤗 Hugging Face Model

**Model Hub**: [fingpt-compliance-agents](https://huggingface.co/QXPS/fingpt-compliance-agents)

The trained model is available on Hugging Face Hub with:
- ✅ Pre-trained LoRA weights
- ✅ Complete model card with performance metrics
- ✅ Usage examples and integration guides
- ✅ Ready-to-use inference scripts

### Quick Start with Hugging Face

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the model directly from Hugging Face
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "QXPS/fingpt-compliance-agents")
tokenizer = AutoTokenizer.from_pretrained("QXPS/fingpt-compliance-agents")

# Use the model
prompt = "Analyze the sentiment of this financial news: 'Company X reported strong quarterly earnings.'"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- UV package manager
- CUDA-compatible GPU (recommended)
- Cloud.ru API key (for audio processing, default)
- OpenAI API key (optional fallback)

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd fingpt-compliance-agents
uv sync
```

2. **Configure the project:**
```bash
cp .env.example .env
# Edit .env with your API keys (Cloud.ru is default for audio)
cp configs/config.yaml.example configs/config.yaml
# Edit configs/config.yaml with your settings
```

3. **Run the complete pipeline:**
```bash
# Full pipeline
make all

# Or use Python directly
uv run python run_pipeline.py --mode lora

# Quick test
uv run python run_pipeline.py --quick
```

## 📁 Project Structure

```
fingpt-compliance-agents/
├── configs/                 # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Data directory
│   ├── raw/               # Raw collected data
│   ├── processed/         # Processed training data
│   ├── train/             # Training data
│   └── test/              # Test data
├── src/                   # Source code
│   ├── data_collection/   # Data collection modules
│   ├── data_processing/   # Data processing modules
│   ├── training/          # Model training modules
│   ├── evaluation/        # Model evaluation modules
│   ├── testing/           # Testing modules
│   └── submission/        # Submission preparation
├── models/                # Trained models
├── logs/                  # Training logs
├── results/               # Evaluation results
├── submission/            # Submission package
├── notebooks/             # Jupyter notebooks
├── Makefile              # Build automation
└── run_pipeline.py       # Main pipeline runner
```

## 🛠 Available Commands

### Setup & Installation
```bash
make setup          # Initial project setup
make install        # Install dependencies
make install-dev    # Install development dependencies
```

### Data Collection
```bash
make data-collect   # Collect all datasets
make data-hf        # Collect Hugging Face datasets only
make data-sec       # Collect SEC EDGAR data only
make data-audio     # Collect financial audio data
```

### Data Processing
```bash
make data-process   # Process and prepare all data
make data-validate  # Validate data quality
```

### Model Training
```bash
make train          # Full training pipeline
make train-lora     # LoRA fine-tuning only
make train-rl       # Reinforcement learning
```

### Evaluation & Testing
```bash
make evaluate       # Evaluate model performance
make test-audio     # Test audio processing
make test-whisper   # Test Cloud.ru Whisper API (default)
make test-whisper-openai # Test OpenAI Whisper API (fallback)
make benchmark      # Run full benchmark suite
```

### Submission
```bash
make prepare-submission  # Prepare for Hugging Face submission
make package            # Create submission package
```

### Utilities
```bash
make clean          # Clean temporary files
make logs           # Show training logs
make status         # Show project status
```

## 📊 Datasets

The model is trained on several financial datasets:

- **FinanceBench**: Financial Q&A on company filings
- **XBRL Analysis**: Tag extraction, value extraction, formula construction
- **Financial Sentiment**: BloombergGPT FPB, FiQA SA datasets
- **SEC EDGAR**: Real SEC filings and XBRL data
- **Financial Audio**: Earnings calls, podcasts, news segments

## 🏗 Model Architecture

- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit quantization (when available)
- **Max Length**: 2048 tokens
- **Parameters**: ~8B (within contest requirements)

## 📈 Performance

The model achieves strong performance on financial tasks:

### Benchmark Results
- **Financial Q&A**: 67.7% accuracy (21/31 correct)
- **XBRL Processing**: 88.3% overall accuracy
  - Tag Extraction: 89.6% accuracy
  - Value Extraction: 63.6% accuracy
  - Formula Construction: 99.4% accuracy
  - Formula Calculation: 82.2% accuracy
- **Sentiment Analysis**: 43.5% accuracy (359/826 correct)
- **Audio Processing**: 100% sentiment accuracy on test samples

### Model Specifications
- **Base Model**: meta-llama/Llama-3.2-1B-Instruct
- **Parameters**: 1.1B (1B base + 0.1B LoRA)
- **Training Data**: 7,153 examples (5,722 train, 1,431 test)
- **Inference Speed**: ~50 tokens/second
- **Memory Usage**: ~4GB VRAM

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

- Model parameters
- Training settings
- Data sources
- Output paths
- API keys

## 📝 Usage Examples

### Basic Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = PeftModel.from_pretrained(base_model, "./models/fingpt-compliance")
tokenizer = AutoTokenizer.from_pretrained("./models/fingpt-compliance")

# Generate response
prompt = "Analyze the sentiment of this financial news: 'Company X reported strong quarterly earnings.'"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using the Pipeline

```python
from run_pipeline import PipelineRunner

# Run complete pipeline
runner = PipelineRunner()
results = runner.run_full_pipeline(mode="lora")

# Run quick test
results = runner.run_quick_test()
```

## 🧪 Testing

Run tests to verify functionality:

```bash
# Run all tests
make test

# Test specific components
uv run python src/testing/audio_tester.py
uv run python src/evaluation/evaluator.py
```

## 📦 Deployment & Submission

### Hugging Face Hub
The model is available on Hugging Face Hub:
- **Model**: [fingpt-compliance-agents](https://huggingface.co/QXPS/fingpt-compliance-agents)
- **Status**: Ready for production use
- **Documentation**: Complete model card with performance metrics

### Local Deployment
```bash
# Prepare submission package
make prepare-submission

# This creates a complete package in the `submission/` directory with:
# - Model weights and configuration
# - Inference scripts
# - Requirements and documentation
# - Evaluation scripts
```

### Cloud Deployment
See [DEPLOYMENT_PLAN.md](DEPLOYMENT_PLAN.md) for detailed deployment options:
- Hugging Face Inference API
- AWS SageMaker
- Google Cloud AI Platform
- Docker containers
- FastAPI applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [FinLoRA](https://github.com/Open-Finance-Lab/FinLoRA) for the fine-tuning framework
- [Hugging Face](https://huggingface.co/) for the model hub and datasets
- [SEC EDGAR](https://www.sec.gov/edgar) for financial filings data
- Contest organizers for providing the challenge and datasets

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the logs in `logs/` directory
- Review the configuration in `configs/config.yaml`

---

**Note**: This project is part of the SecureFinAI Contest 2025 - Task 2: FinGPT-Powered Compliance Agents