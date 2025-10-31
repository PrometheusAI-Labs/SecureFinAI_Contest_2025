#!/usr/bin/env python3
"""
FinGPT Compliance Agents Trainer
Handles model training with LoRA fine-tuning and reinforcement learning
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FinGPTDataset(Dataset):
    """Dataset class for FinGPT training data"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        conversations = example['conversations']
        
        # Format conversations for training
        text = self.format_conversations(conversations)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }
    
    def format_conversations(self, conversations: List[Dict]) -> str:
        """Format conversations into training text"""
        formatted = ""
        
        for turn in conversations:
            role = turn['role']
            content = turn['content']
            
            if role == 'system':
                formatted += f"<|system|>\n{content}\n"
            elif role == 'user':
                formatted += f"<|user|>\n{content}\n"
            elif role == 'assistant':
                formatted += f"<|assistant|>\n{content}\n"
        
        formatted += "<|endoftext|>"
        return formatted

class FinGPTTrainer:
    """Main trainer class for FinGPT Compliance Agents"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.lora_config = self.config['lora']
        self.training_config = self.config['training']
        
        # Setup paths
        self.data_dir = Path(self.config['output']['data_path'])
        self.model_dir = Path(self.config['output']['model_path'])
        self.logs_dir = Path(self.config['output']['logs_path'])
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.logs_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
    
    def setup_model(self):
        """Setup model and tokenizer"""
        logger.info("Setting up model and tokenizer")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model'],
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if available
        try:
            # Try to use BitsAndBytesConfig for quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['base_model'],
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        except Exception as e:
            logger.warning(f"Quantization not available: {e}")
            # Fallback to regular model loading
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['base_model'],
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied")
    
    def load_training_data(self) -> tuple:
        """Load training and validation data"""
        logger.info("Loading training data")
        
        # Load train data
        train_file = self.data_dir / 'train' / 'train.jsonl'
        test_file = self.data_dir / 'test' / 'test.jsonl'
        
        train_data = []
        test_data = []
        
        # Load train data
        if train_file.exists():
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line.strip()))
        
        # Load test data
        if test_file.exists():
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    test_data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(train_data)} train examples, {len(test_data)} test examples")
        
        return train_data, test_data
    
    def create_datasets(self, train_data: List[Dict], test_data: List[Dict]) -> tuple:
        """Create PyTorch datasets"""
        logger.info("Creating datasets")
        
        train_dataset = FinGPTDataset(
            train_data, 
            self.tokenizer, 
            self.model_config['max_length']
        )
        
        test_dataset = FinGPTDataset(
            test_data, 
            self.tokenizer, 
            self.model_config['max_length']
        )
        
        return train_dataset, test_dataset
    
    def train(self, mode: str = "full"):
        """Main training function"""
        logger.info(f"Starting training in {mode} mode")
        
        # Setup model
        self.setup_model()
        
        if mode in ["lora", "full"]:
            self.setup_lora()
        
        # Load data
        train_data, test_data = self.load_training_data()
        
        if not train_data:
            logger.error("No training data found!")
            return
        
        # Create datasets
        train_dataset, test_dataset = self.create_datasets(train_data, test_data)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(self.model_dir),
            num_train_epochs=self.training_config['num_epochs'],
            per_device_train_batch_size=self.training_config['batch_size'],
            per_device_eval_batch_size=self.training_config['batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=float(self.training_config['learning_rate']),
            warmup_steps=self.training_config['warmup_steps'],
            logging_steps=100,
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model if mode in ["lora", "full"] else self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            processing_class=self.tokenizer,
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.model_dir)
        
        # Save training config
        with open(self.model_dir / 'training_config.json', 'w') as f:
            json.dump(self.training_config, f, indent=2)
        
        logger.info("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Train FinGPT Compliance Agents')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['lora', 'full'], default='lora', help='Training mode')
    
    args = parser.parse_args()
    
    trainer = FinGPTTrainer(args.config)
    trainer.train(args.mode)

if __name__ == "__main__":
    main()