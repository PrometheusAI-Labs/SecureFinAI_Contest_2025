# Quick Start Guide

## Prerequisites

- Python 3.10+
- UV package manager
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/xsa-dev/SecureFinAI_Contest_2025.git
   cd SecureFinAI_Contest_2025/Task_1_FinRL_DT_Crypto_Trading
   ```

2. **Install dependencies**
   ```bash
   make install
   ```

3. **Download datasets**
   ```bash
   make setup-data
   ```

## Quick Start

### Option 1: Complete Workflow
Run the entire pipeline from data preparation to model evaluation:
```bash
make workflow
```

### Option 2: Step-by-Step
Execute individual steps as needed:
```bash
make step1    # Generate Alpha101 factors
make step2    # Train RNN factor aggregation
make step3    # Train single RL agent
make step4    # Train ensemble RL agents
make step5    # Convert RL trajectories
make step6    # Train Decision Transformer
make step7    # Evaluate Decision Transformer
```

## Key Commands

```bash
make help              # Show all available commands
make status            # Check project status
make check-data        # Verify data integrity
make train-dt          # Train Decision Transformer
make evaluate-dt       # Evaluate trained model
make clean             # Clean temporary files
```

## Project Structure

```
├── Makefile                           # Main build file
├── dt_crypto.py                       # Decision Transformer training
├── evaluation.py                      # Model evaluation
├── download_data.py                   # Data download script
├── offline_data_preparation/          # Data preprocessing
│   ├── seq_data.py                   # Alpha101 factor generation
│   ├── seq_run.py                    # RNN training
│   ├── erl_run.py                    # Single RL agent
│   ├── task1_ensemble.py             # Ensemble RL agents
│   └── convert_replay_buffer_to_trajectories.py
├── trained_models/                    # Saved models
├── plots/                            # Generated plots
└── data/                             # Datasets
```

## Pre-trained Models and Results

This repository includes pre-trained models and evaluation results:

### Trained Models

1. **Decision Transformer** (`trained_models/decision_transformer.pth`)
   - Final trained model for crypto trading
   - Size: ~1.9MB

2. **Ensemble RL Models** (`offline_data_preparation/ensemble_teamname/ensemble_models/`)
   - AgentD3QN, AgentDoubleDQN, AgentTwinD3QN
   - Each agent includes: actor, critic, target networks, and optimizers
   - Total size: ~5.1MB

3. **Individual RL Agents** (`offline_data_preparation/TradeSimulator-v0_*/`)
   - TradeSimulator-v0_D3QN_0/ (D3QN agent)
   - TradeSimulator-v0_DoubleDQN_1000/ (DoubleDQN, seed 1000)
   - TradeSimulator-v0_DoubleDQN_2000/ (DoubleDQN, seed 2000)
   - TradeSimulator-v0_TwinD3QN_3000/ (TwinD3QN, seed 3000)
   - Each directory contains model weights (~2MB)

**Note**: Replay buffers can be regenerated from model weights using `generate_replay_buffer_from_model.py` if needed.

### Evaluation Plots

The `plots/` directory contains visualization results:

1. **crypto_evaluation_comprehensive.png** (843KB)
   - Comprehensive evaluation metrics and performance analysis

2. **crypto_portfolio_comparison.png** (317KB)
   - Portfolio value comparison and benchmark analysis

3. **training_loss_plot_crypto.png** (47KB)
   - Training loss curves during Decision Transformer training

## Data Sources

- **FinRL_BTC_news_signals**: [Hugging Face](https://huggingface.co/datasets/SecureFinAI-Lab/FinRL_BTC_news_signals)
- **BTC_1sec_with_sentiment_risk_train.csv**: [Google Drive](https://drive.google.com/drive/folders/1rV9tJ0T2iWNJ-g3TI4Qgqy0cVf_Zqzqp?usp=sharing)

## Troubleshooting

1. **Check project status**: `make status`
2. **Verify data**: `make check-data`
3. **Clean and restart**: `make clean && make workflow`
4. **View detailed usage**: `cat MAKEFILE_USAGE.md`

## Reporting Issues

If you find any inaccuracies, bugs, or have suggestions for improvement, please create an issue in the repository:

- **GitHub Issues**: [Create an issue](https://github.com/xsa-dev/SecureFinAI_Contest_2025/issues/new)
- Include details about:
  - What you were trying to do
  - What happened (or what you expected)
  - Steps to reproduce (if applicable)
  - Environment details (Python version, OS, etc.)

## Next Steps

After successful setup:
1. Review the generated plots in `plots/` directory
2. Check model performance metrics
3. Experiment with different hyperparameters
4. Analyze trading strategies and results

For detailed information, see the main [README.md](README.md) and [MAKEFILE_USAGE.md](MAKEFILE_USAGE.md).