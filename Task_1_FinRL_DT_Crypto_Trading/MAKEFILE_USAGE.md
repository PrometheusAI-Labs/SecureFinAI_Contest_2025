# Makefile Usage Guide

## Quick Start

### 1. Complete Setup and Workflow Execution
```bash
make workflow
```
This will execute all steps: dependency installation, data downloading, factor generation, RL agent training, trajectory conversion, Decision Transformer training and evaluation.

### 2. Step-by-Step Execution

#### Step 1: Installation and Data
```bash
make install          # Install dependencies
make setup-data       # Download datasets
make check-data       # Verify data integrity
```

#### Step 2: Data Preparation
```bash
make step1            # Generate Alpha101 factors
make step2            # Train RNN for factor aggregation
```

#### Step 3-4: RL Agent Training
```bash
make step3            # Train single RL agent
make step4            # Train ensemble RL agents
make step5            # Convert trajectories for DT
```

#### Step 5-6: Decision Transformer
```bash
make step6            # Train Decision Transformer
make step7            # Evaluate Decision Transformer
```

## Main Commands

### Data Management
```bash
make download-data    # Download data from Hugging Face and Google Drive
make clean-data       # Clean downloaded data files
make check-data       # Verify data integrity
```

### Model Training
```bash
make train-dt         # Train Decision Transformer (main model)
make train-ensemble   # Train ensemble RL agents
make train-rl         # Train single RL agent
make train-factors    # Train RNN for factors
```

### Evaluation and Testing
```bash
make evaluate-dt      # Evaluate Decision Transformer
make test             # Run tests
```

### Development
```bash
make format           # Format code
make lint             # Run code linting
make check-deps       # Check dependencies
```

### Project Management
```bash
make status           # Show project and data status
make clean            # Clean temporary files
make help             # Show all available commands
```

## Workflow Structure

1. **Data Preparation** - Data preprocessing
   - Dataset downloading
   - Alpha101 technical factor generation
   - RNN training for factor aggregation

2. **RL Training** - RL agent training
   - Single RL agent training
   - Ensemble RL agent training
   - Replay buffer to trajectory conversion

3. **Decision Transformer** - DT training
   - Decision Transformer model training
   - Performance evaluation

## Default Parameters

### Decision Transformer
- Epochs: 100
- Learning rate: 1e-3
- Context length: 20
- Model path: `./trained_models/decision_transformer.pth`
- Plots directory: `plots`

### Evaluation
- Max samples: 35000
- Target return: 250.0
- Context length: 20

## Troubleshooting

### Check Status
```bash
make status
```

### Clean and Restart
```bash
make clean
make workflow
```

### Verify Data
```bash
make check-data
make analyze-news
make analyze-trading
```