# Makefile for FinAI Contest Task 1 - Crypto Trading with Decision Transformer
# SecureFinAI Contest 2025 - Task 1

.PHONY: help install setup-data download-data clean-data train evaluate test format lint check-deps

# Default target
help:
	@echo "FinAI Contest Task 1 - Crypto Trading with Decision Transformer"
	@echo "Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install project dependencies using uv"
	@echo "  setup-data       Download and prepare all required datasets"
	@echo "  download-data    Download datasets from Hugging Face and Google Drive"
	@echo ""
	@echo "Data Preparation (Step 1-2):"
	@echo "  prepare-data     Generate Alpha101 technical factors from BTC data"
	@echo "  create-test-split Create train/test data split (20% test)"
	@echo "  train-factors    Train RNN models to aggregate factor sequences"
	@echo ""
	@echo "RL Training (Step 3-4):"
	@echo "  train-rl         Train single RL agent"
	@echo "  train-ensemble   Train ensemble RL agents"
	@echo "  convert-traj     Convert RL replay buffers to DT trajectories"
	@echo ""
	@echo "Decision Transformer (Step 5-6):"
	@echo "  train-dt         Train Decision Transformer model"
	@echo "  evaluate-dt      Evaluate trained Decision Transformer"
	@echo ""
	@echo "Data Management:"
	@echo "  clean-data       Clean downloaded data files"
	@echo "  check-data       Verify data integrity and structure"
	@echo "  check-dependencies Check workflow dependencies"
	@echo ""
	@echo "Development:"
	@echo "  format           Format code using black and isort"
	@echo "  lint             Run linting checks"
	@echo "  check-deps       Check for outdated dependencies"
	@echo ""
	@echo "Project Management:"
	@echo "  clean            Clean all generated files"
	@echo "  status           Show project status and data info"
	@echo "  workflow         Run complete workflow (data -> factors -> RL -> DT)"
	@echo ""
	@echo "Individual Steps:"
	@echo "  step1            Generate Alpha101 factors"
	@echo "  step1.5          Create train/test data split"
	@echo "  step2            Train RNN factor aggregation"
	@echo "  step3            Train single RL agent"
	@echo "  step4            Train ensemble RL agents"
	@echo "  step5            Convert RL trajectories"
	@echo "  step6            Train Decision Transformer"
	@echo "  step7            Evaluate Decision Transformer"

# Installation and setup
install:
	@echo "Installing project dependencies..."
	uv sync
	@echo "Dependencies installed successfully!"

setup-data: download-data check-data
	@echo "Data setup completed!"

download-data:
	@echo "Downloading datasets..."
	uv run python download_data.py
	@echo "Data download completed!"

# Data management
clean-data:
	@echo "Cleaning data files..."
	rm -f offline_data_preparation/data/*.part
	rm -f offline_data_preparation/data/FinRL_BTC_news_signals.csv
	@echo "Data files cleaned!"

check-dependencies:
	@echo "Checking workflow dependencies..."
	@echo "Step 1 - Data preparation:"
	@if [ -f "offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv" ]; then \
		echo "  ✅ BTC training data found"; \
	else \
		echo "  ❌ BTC training data missing - run 'make download-data'"; \
	fi
	@if [ -f "offline_data_preparation/data/BTC_1sec_with_sentiment_risk_test.csv" ]; then \
		echo "  ✅ BTC test data found"; \
	else \
		echo "  ❌ BTC test data missing - run 'make create-test-split'"; \
	fi
	@if [ -f "offline_data_preparation/data/FinRL_BTC_news_signals.csv" ]; then \
		echo "  ✅ News data found"; \
	else \
		echo "  ❌ News data missing - run 'make download-data'"; \
	fi
	@echo "Step 2 - Alpha101 factors:"
	@if [ -f "offline_data_preparation/data/BTC_1sec_input.npy" ] && [ -f "offline_data_preparation/data/BTC_1sec_label.npy" ]; then \
		echo "  ✅ Alpha101 factors found"; \
	else \
		echo "  ❌ Alpha101 factors missing - run 'make prepare-data'"; \
	fi
	@echo "Step 3-4 - RL training:"
	@if [ -f "offline_data_preparation/TradeSimulator-v0_D3QN_-1/replay_buffer_states.pth" ]; then \
		echo "  ✅ RL training output found"; \
	else \
		echo "  ❌ RL training output missing - run 'make train-ensemble'"; \
	fi
	@echo "Step 5 - Trajectories:"
	@if [ -f "crypto_trajectories.csv" ]; then \
		echo "  ✅ Trajectory data found"; \
	else \
		echo "  ❌ Trajectory data missing - run 'make convert-traj'"; \
	fi
	@echo "Step 6 - Decision Transformer:"
	@if [ -f "trained_models/decision_transformer.pth" ]; then \
		echo "  ✅ Decision Transformer model found"; \
	else \
		echo "  ❌ Decision Transformer model missing - run 'make train-dt'"; \
	fi

check-data:
	@echo "Checking data integrity..."
	@if [ -f "offline_data_preparation/data/FinRL_BTC_news_signals.csv" ]; then \
		echo "✅ FinRL_BTC_news_signals.csv found"; \
		uv run python -c "import pandas as pd; df=pd.read_csv('offline_data_preparation/data/FinRL_BTC_news_signals.csv'); print(f'   Shape: {df.shape}'); print(f'   Columns: {list(df.columns)}')"; \
	else \
		echo "❌ FinRL_BTC_news_signals.csv not found"; \
	fi
	@if [ -f "offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv" ]; then \
		echo "✅ BTC_1sec_with_sentiment_risk_train.csv found"; \
		uv run python -c "import pandas as pd; df=pd.read_csv('offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv', nrows=5); print(f'   Columns: {list(df.columns)}')"; \
	else \
		echo "❌ BTC_1sec_with_sentiment_risk_train.csv not found"; \
	fi

# Data Preparation (Step 1-2)
prepare-data:
	@echo "Step 1: Generating Alpha101 technical factors from BTC data..."
	cd offline_data_preparation && uv run python seq_data.py
	@echo "Alpha101 factors generated!"

create-test-split:
	@echo "Step 1.5: Creating train/test data split..."
	@if [ ! -f "offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv" ]; then \
		echo "❌ Training data not found. Please run 'make download-data' first."; \
		exit 1; \
	fi
	uv run python create_test_split.py --input_file offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv --output_dir offline_data_preparation/data --test_size 0.2
	@echo "Test data split created!"

train-factors:
	@echo "Step 2: Training RNN models to aggregate factor sequences..."
	@if [ ! -f "offline_data_preparation/data/BTC_1sec_input.npy" ] || [ ! -f "offline_data_preparation/data/BTC_1sec_label.npy" ]; then \
		echo "❌ Alpha101 factors not found. Please run 'make prepare-data' first."; \
		exit 1; \
	fi
	cd offline_data_preparation && uv run python seq_run.py 0
	@echo "RNN factor aggregation training completed!"

# RL Training (Step 3-4)
train-rl:
	@echo "Step 3: Training single RL agent..."
	cd offline_data_preparation && uv run python erl_run.py 0
	@echo "Single RL agent training completed!"

train-ensemble:
	@echo "Step 4: Training ensemble RL agents..."
	cd offline_data_preparation && uv run python task1_ensemble.py 0
	@echo "Ensemble RL training completed!"

convert-traj:
	@echo "Step 5: Converting RL replay buffers to DT trajectories..."
	@if [ ! -f "offline_data_preparation/TradeSimulator-v0_D3QN_-1/replay_buffer_states.pth" ]; then \
		echo "❌ RL training files not found. Please run 'make train-ensemble' first."; \
		exit 1; \
	fi
	cd offline_data_preparation && uv run python convert_replay_buffer_to_trajectories.py --replay_buffer_dir ./TradeSimulator-v0_D3QN_-1 --output_file ../crypto_trajectories.csv
	@echo "Trajectory conversion completed!"

# Decision Transformer (Step 5-6)
train-dt:
	@echo "Step 6: Training Decision Transformer model..."
	@if [ ! -f "crypto_trajectories.csv" ]; then \
		echo "❌ Trajectory data not found. Please run 'make convert-traj' first."; \
		exit 1; \
	fi
	uv run python dt_crypto.py --epochs 100 --lr 1e-3 --context_length 20 --model_path ./trained_models/decision_transformer.pth --plots_dir plots
	@echo "Decision Transformer training completed!"

evaluate-dt:
	@echo "Evaluating Decision Transformer model..."
	@if [ ! -f "offline_data_preparation/data/BTC_1sec_with_sentiment_risk_test.csv" ]; then \
		echo "❌ Test data not found. Please run 'make create-test-split' first."; \
		exit 1; \
	fi
	uv run python evaluation.py --model_path ./trained_models/decision_transformer.pth --test_data_path ./offline_data_preparation/data/BTC_1sec_with_sentiment_risk_test.csv --max_samples 35000 --target_return 250.0 --context_length 20 --plots_dir plots
	@echo "Decision Transformer evaluation completed!"

# Legacy commands for backward compatibility
train: train-dt
evaluate: evaluate-dt

test:
	@echo "Running tests..."
	uv run python -m pytest tests/ -v
	@echo "Tests completed!"

# Development tools
format:
	@echo "Formatting code..."
	uv run black .
	uv run isort .
	@echo "Code formatted!"

lint:
	@echo "Running linting checks..."
	uv run flake8 .
	uv run mypy .
	@echo "Linting completed!"

check-deps:
	@echo "Checking for outdated dependencies..."
	uv tree
	@echo "Dependency check completed!"

# Project management
clean:
	@echo "Cleaning project..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.part" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "Project cleaned!"

status:
	@echo "=== Project Status ==="
	@echo "Python version: $$(uv run python --version)"
	@echo "UV version: $$(uv --version)"
	@echo ""
	@echo "=== Data Status ==="
	@make check-data
	@echo ""
	@echo "=== Model Status ==="
	@if [ -f "trained_models/decision_transformer.pth" ]; then \
		echo "✅ Decision Transformer model found"; \
		ls -lh trained_models/; \
	else \
		echo "❌ No trained model found"; \
	fi
	@echo ""
	@echo "=== Project Structure ==="
	@echo "Main files:"
	@ls -la *.py *.md Makefile 2>/dev/null || true
	@echo ""
	@echo "Data directory:"
	@ls -la offline_data_preparation/data/ 2>/dev/null || true

# Complete workflow
workflow: install setup-data create-test-split prepare-data train-factors train-ensemble convert-traj train-dt evaluate-dt
	@echo "Complete workflow finished! Check results in plots/ directory."

# Quick development commands
dev-setup: install setup-data
	@echo "Development environment ready!"

quick-train: check-data train-dt
	@echo "Quick Decision Transformer training completed!"

# Individual step commands
step1: prepare-data
step1.5: create-test-split
step2: train-factors
step3: train-rl
step4: train-ensemble
step5: convert-traj
step6: train-dt
step7: evaluate-dt

# Data analysis commands
analyze-news:
	@echo "Analyzing news data..."
	uv run python -c "import pandas as pd; df=pd.read_csv('offline_data_preparation/data/FinRL_BTC_news_signals.csv'); print('News Data Analysis:'); print(f'Total news: {len(df)}'); print(f'Sentiment distribution:'); print(df['sentiment_score'].value_counts().sort_index()); print(f'Risk distribution:'); print(df['risk_score'].value_counts().sort_index())"

analyze-trading:
	@echo "Analyzing trading data..."
	uv run python -c "import pandas as pd; df=pd.read_csv('offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv', nrows=1000); print('Trading Data Analysis:'); print(f'Columns: {list(df.columns)}'); print(f'Sample data:'); print(df.head())"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Data Download Report:"
	@cat data_download_report.md
	@echo ""
	@echo "Project README:"
	@cat README.md