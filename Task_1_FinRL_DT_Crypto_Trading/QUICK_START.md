# Quick Start Guide

## Prerequisites

- Python 3.10+
- UV package manager
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Task_1_FinRL_DT_Crypto_Trading
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

## Data Sources

- **FinRL_BTC_news_signals**: [Hugging Face](https://huggingface.co/datasets/SecureFinAI-Lab/FinRL_BTC_news_signals)
- **BTC_1sec_with_sentiment_risk_train.csv**: [Google Drive](https://drive.google.com/drive/folders/1rV9tJ0T2iWNJ-g3TI4Qgqy0cVf_Zqzqp?usp=sharing)

## Troubleshooting

1. **Check project status**: `make status`
2. **Verify data**: `make check-data`
3. **Clean and restart**: `make clean && make workflow`
4. **View detailed usage**: `cat MAKEFILE_USAGE.md`

## Next Steps

After successful setup:
1. Review the generated plots in `plots/` directory
2. Check model performance metrics
3. Experiment with different hyperparameters
4. Analyze trading strategies and results

For detailed information, see the main [README.md](README.md) and [MAKEFILE_USAGE.md](MAKEFILE_USAGE.md).