# FinGPT Compliance Agents - Main Pipeline
# Makefile for automating the entire task execution pipeline

.PHONY: help setup install data-collect data-process train evaluate test-audio prepare-submission clean all

# Default target
help:
	@echo "FinGPT Compliance Agents - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup          - Initial project setup and environment creation"
	@echo "  install        - Install all dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo ""
	@echo "Data Collection:"
	@echo "  data-collect   - Collect all required datasets"
	@echo "  data-sec       - Collect SEC EDGAR data only"
	@echo "  data-hf        - Collect Hugging Face datasets only"
	@echo "  data-audio     - Collect financial audio data"
	@echo "  data-xbrl      - Collect XBRL specifications and documentation"
	@echo ""
	@echo "Data Processing:"
	@echo "  data-process   - Process and prepare all collected data"
	@echo "  data-validate  - Validate data quality and format"
	@echo ""
	@echo "Model Training:"
	@echo "  train          - Train the FinGPT compliance model"
	@echo "  train-lora     - Train with LoRA fine-tuning only"
	@echo "  train-rl       - Train with reinforcement learning"
	@echo "  train-and-evaluate - Train model and then evaluate it"
	@echo ""
	@echo "Evaluation & Testing:"
	@echo "  evaluate       - Evaluate model on all tasks (requires trained model)"
	@echo "  test-audio     - Test audio processing capabilities"
	@echo "  test-whisper   - Test Cloud.ru Whisper API connection"
	@echo "  test-whisper-openai - Test OpenAI Whisper API connection"
	@echo "  benchmark      - Run full benchmark suite"
	@echo ""
	@echo "Submission & Deployment:"
	@echo "  prepare-submission - Prepare model for Hugging Face submission"
	@echo "  package        - Create submission package"
	@echo "  upload-hf      - Upload model to Hugging Face Hub"
	@echo "  deploy-cloud   - Deploy model to cloud platforms"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          - Clean temporary files and caches"
	@echo "  logs           - Show training logs"
	@echo "  status         - Show project status"
	@echo ""
	@echo "Full Pipeline:"
	@echo "  all            - Run complete pipeline (setup -> train -> evaluate -> prepare-submission)"

# Configuration
PYTHON := uv run python
CONFIG_FILE := configs/config.yaml
DATA_DIR := data
MODELS_DIR := models
LOGS_DIR := logs
RESULTS_DIR := results

# Setup and Installation
setup: install
	@echo "Setting up project structure..."
	@mkdir -p $(DATA_DIR)/{raw,processed,train,test}
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p notebooks
	@echo "Project structure created!"

install:
	@echo "Installing dependencies..."
	@uv sync
	@echo "Dependencies installed!"

install-dev: install
	@echo "Installing development dependencies..."
	@uv add --dev pytest black flake8 mypy jupyter
	@echo "Development dependencies installed!"

# Data Collection
data-collect: data-hf data-sec data-audio data-xbrl
	@echo "All data collection completed!"

data-hf:
	@echo "Collecting Hugging Face datasets..."
	@$(PYTHON) src/data_collection/hf_data_collector.py
	@echo "Hugging Face datasets collected!"

data-sec:
	@echo "Collecting SEC EDGAR data..."
	@$(PYTHON) src/data_collection/sec_data_collector.py
	@echo "SEC EDGAR data collected!"

data-audio:
	@echo "Collecting financial audio data..."
	@$(PYTHON) src/data_collection/audio_data_collector.py
	@echo "Financial audio data collected!"

data-xbrl:
	@echo "Collecting XBRL specifications and documentation..."
	@$(PYTHON) src/data_collection/xbrl_data_collector.py --max-pages 30
	@echo "XBRL data collected!"

# Data Processing
data-process:
	@echo "Processing collected data..."
	@$(PYTHON) src/data_processing/data_processor.py --config $(CONFIG_FILE)
	@echo "Data processing completed!"

data-validate:
	@echo "Validating data quality..."
	@$(PYTHON) src/data_processing/data_validator.py --config $(CONFIG_FILE)
	@echo "Data validation completed!"

# Model Training
train: data-process
	@echo "Starting model training..."
	@$(PYTHON) src/training/trainer.py --config $(CONFIG_FILE) --mode full
	@echo "Model training completed!"

train-lora: data-process
	@echo "Starting LoRA fine-tuning..."
	@$(PYTHON) src/training/trainer.py --config $(CONFIG_FILE) --mode lora
	@echo "LoRA fine-tuning completed!"

train-rl: train-lora
	@echo "Starting reinforcement learning training..."
	@$(PYTHON) src/training/rl_trainer.py --config $(CONFIG_FILE)
	@echo "Reinforcement learning training completed!"

# Evaluation and Testing
evaluate:
	@echo "Evaluating model performance..."
	@$(PYTHON) src/evaluation/evaluator.py --config $(CONFIG_FILE)
	@echo "Model evaluation completed!"

train-and-evaluate: train evaluate
	@echo "Training and evaluation completed!"

test-audio:
	@echo "Testing audio processing capabilities..."
	@$(PYTHON) src/testing/audio_tester.py --config $(CONFIG_FILE)
	@echo "Audio processing test completed!"

test-whisper:
	@echo "Testing Cloud.ru Whisper API connection..."
	@$(PYTHON) test_whisper_connection.py --provider cloud_ru --test-audio
	@echo "Cloud.ru Whisper API test completed!"

test-whisper-openai:
	@echo "Testing OpenAI Whisper API connection..."
	@$(PYTHON) test_whisper_connection.py --provider openai --test-audio
	@echo "OpenAI Whisper API test completed!"

benchmark: evaluate test-audio
	@echo "Running full benchmark suite..."
	@$(PYTHON) src/evaluation/benchmark.py --config $(CONFIG_FILE)
	@echo "Benchmark completed!"

# Submission Preparation
prepare-submission: evaluate
	@echo "Preparing model for submission..."
	@$(PYTHON) src/submission/prepare_submission.py --config $(CONFIG_FILE)
	@echo "Submission preparation completed!"

package: prepare-submission
	@echo "Creating submission package..."
	@$(PYTHON) src/submission/package_model.py --config $(CONFIG_FILE)
	@echo "Submission package created!"

upload-hf: prepare-submission
	@echo "Uploading model to Hugging Face Hub..."
	@$(PYTHON) upload_to_huggingface.py --model-path ./models/fingpt-compliance
	@echo "Model uploaded to Hugging Face Hub!"

deploy-cloud: upload-hf
	@echo "Deploying model to cloud platforms..."
	@echo "Model is now available on Hugging Face Hub"
	@echo "See DEPLOYMENT_PLAN.md for additional cloud deployment options"
	@echo "Cloud deployment completed!"

# Utilities
clean:
	@echo "Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf $(LOGS_DIR)/*.log
	@echo "Cleanup completed!"

logs:
	@echo "Recent training logs:"
	@tail -n 50 $(LOGS_DIR)/*.log 2>/dev/null || echo "No logs found"

status:
	@echo "Project Status:"
	@echo "==============="
	@echo "Data directory: $(DATA_DIR)"
	@ls -la $(DATA_DIR)/ 2>/dev/null || echo "Data directory not found"
	@echo ""
	@echo "Models directory: $(MODELS_DIR)"
	@ls -la $(MODELS_DIR)/ 2>/dev/null || echo "Models directory not found"
	@echo ""
	@echo "Logs directory: $(LOGS_DIR)"
	@ls -la $(LOGS_DIR)/ 2>/dev/null || echo "Logs directory not found"

# Full Pipeline
all: setup data-collect data-process train evaluate prepare-submission
	@echo "Complete pipeline executed successfully!"
	@echo "Model ready for submission to Hugging Face"

# Development helpers
dev-setup: install-dev
	@echo "Setting up development environment..."
	@$(PYTHON) -m ipykernel install --user --name fingpt-compliance
	@echo "Development environment ready!"

test:
	@echo "Running tests..."
	@$(PYTHON) -m pytest tests/ -v

lint:
	@echo "Running linters..."
	@$(PYTHON) -m flake8 src/
	@$(PYTHON) -m black --check src/
	@$(PYTHON) -m mypy src/

format:
	@echo "Formatting code..."
	@$(PYTHON) -m black src/
	@echo "Code formatted!"

# Quick start for testing
quick-test: setup data-hf data-process train-lora evaluate
	@echo "Quick test pipeline completed!"

# Production deployment
deploy: all package
	@echo "Model deployed and ready for production!"

# Help for specific targets
help-data:
	@echo "Data Collection Commands:"
	@echo "  data-collect   - Collect all datasets"
	@echo "  data-hf        - Collect Hugging Face datasets"
	@echo "  data-sec       - Collect SEC EDGAR data"
	@echo "  data-audio     - Collect audio data"

help-train:
	@echo "Training Commands:"
	@echo "  train          - Full training pipeline"
	@echo "  train-lora     - LoRA fine-tuning only"
	@echo "  train-rl       - Reinforcement learning"

help-eval:
	@echo "Evaluation Commands:"
	@echo "  evaluate       - Evaluate model performance"
	@echo "  test-audio     - Test audio processing"
	@echo "  benchmark      - Run full benchmark suite"