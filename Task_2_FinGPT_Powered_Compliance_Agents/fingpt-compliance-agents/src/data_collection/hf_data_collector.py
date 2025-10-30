"""
Hugging Face Data Collector
Collects financial datasets from Hugging Face Hub
"""

import os
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HFDataCollector:
    """Collects financial datasets from Hugging Face Hub"""
    
    def __init__(self, output_dir: str = "data/raw/hf"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets_config = {
            'financebench': {
                'name': 'PatronusAI/financebench',
                'description': 'Financial Q&A on company filings',
                'tasks': ['financial_qa']
            },
            'xbrl_analysis': {
                'name': 'wangd12/XBRL_analysis',
                'description': 'XBRL tag and value extraction',
                'tasks': ['xbrl_tag_extraction', 'xbrl_value_extraction', 'xbrl_formula_construction', 'xbrl_formula_calculation']
            },
            'fpb_sentiment': {
                'name': 'ChanceFocus/en-fpb',
                'description': 'Financial sentiment analysis',
                'tasks': ['sentiment_analysis']
            },
            'fia_sentiment': {
                'name': 'ChanceFocus/en-fia',
                'description': 'Financial sentiment analysis (FiQA)',
                'tasks': ['sentiment_analysis']
            }
        }
    
    def load_dataset(self, dataset_name: str, split: str = 'train') -> Optional[Dataset]:
        """Load a dataset from Hugging Face Hub"""
        if dataset_name not in self.datasets_config:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        try:
            logger.info(f"Loading dataset: {self.datasets_config[dataset_name]['name']}")
            dataset = load_dataset(
                self.datasets_config[dataset_name]['name'],
                split=split
            )
            logger.info(f"Loaded {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return None
    
    def save_dataset(self, dataset: Dataset, dataset_name: str, split: str = 'train') -> str:
        """Save dataset to local files"""
        output_path = self.output_dir / f"{dataset_name}_{split}.jsonl"
        
        # Convert to JSONL format
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(dataset)} samples to {output_path}")
        return str(output_path)
    
    def process_financebench(self) -> Dict[str, str]:
        """Process FinanceBench dataset for financial Q&A"""
        logger.info("Processing FinanceBench dataset")
        
        # Load train and test splits
        train_dataset = self.load_dataset('financebench', 'train')
        test_dataset = self.load_dataset('financebench', 'test')
        
        saved_files = {}
        
        if train_dataset:
            train_path = self.save_dataset(train_dataset, 'financebench', 'train')
            saved_files['train'] = train_path
        
        if test_dataset:
            test_path = self.save_dataset(test_dataset, 'financebench', 'test')
            saved_files['test'] = test_path
        
        return saved_files
    
    def process_xbrl_analysis(self) -> Dict[str, str]:
        """Process XBRL Analysis dataset"""
        logger.info("Processing XBRL Analysis dataset")
        
        # Load the dataset
        dataset = self.load_dataset('xbrl_analysis')
        
        if not dataset:
            return {}
        
        saved_files = {}
        
        # Process different splits if available
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                split_path = self.save_dataset(dataset[split_name], 'xbrl_analysis', split_name)
                saved_files[split_name] = split_path
        
        return saved_files
    
    def process_sentiment_datasets(self) -> Dict[str, str]:
        """Process financial sentiment analysis datasets"""
        logger.info("Processing sentiment analysis datasets")
        
        saved_files = {}
        
        # Process FPB dataset
        fpb_train = self.load_dataset('fpb_sentiment', 'train')
        fpb_test = self.load_dataset('fpb_sentiment', 'test')
        
        if fpb_train:
            fpb_train_path = self.save_dataset(fpb_train, 'fpb_sentiment', 'train')
            saved_files['fpb_train'] = fpb_train_path
        
        if fpb_test:
            fpb_test_path = self.save_dataset(fpb_test, 'fpb_sentiment', 'test')
            saved_files['fpb_test'] = fpb_test_path
        
        # Process FiQA dataset
        fia_train = self.load_dataset('fia_sentiment', 'train')
        fia_test = self.load_dataset('fia_sentiment', 'test')
        
        if fia_train:
            fia_train_path = self.save_dataset(fia_train, 'fia_sentiment', 'train')
            saved_files['fia_train'] = fia_train_path
        
        if fia_test:
            fia_test_path = self.save_dataset(fia_test, 'fia_sentiment', 'test')
            saved_files['fia_test'] = fia_test_path
        
        return saved_files
    
    def create_training_data(self, dataset_files: Dict[str, str]) -> str:
        """Create unified training data from collected datasets"""
        logger.info("Creating unified training data")
        
        all_data = []
        
        # Process each dataset file
        for dataset_name, file_path in dataset_files.items():
            if not os.path.exists(file_path):
                continue
            
            logger.info(f"Processing {dataset_name} from {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        # Convert to standard format
                        if 'question' in data and 'answer' in data:
                            # FinanceBench format
                            formatted_data = {
                                'context': data['question'],
                                'target': data['answer'],
                                'source': 'financebench',
                                'task': 'financial_qa'
                            }
                        elif 'text' in data and 'gold' in data:
                            # Sentiment analysis format
                            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                            formatted_data = {
                                'context': data['text'],
                                'target': label_map.get(data['gold'], 'neutral'),
                                'source': 'sentiment',
                                'task': 'sentiment_analysis'
                            }
                        elif 'input' in data and 'output' in data:
                            # XBRL format
                            formatted_data = {
                                'context': data['input'],
                                'target': data['output'],
                                'source': 'xbrl',
                                'task': 'xbrl_analysis'
                            }
                        else:
                            # Generic format
                            formatted_data = {
                                'context': str(data.get('context', data.get('question', data.get('text', '')))),
                                'target': str(data.get('target', data.get('answer', data.get('output', '')))),
                                'source': dataset_name,
                                'task': 'general'
                            }
                        
                        if formatted_data['context'] and formatted_data['target']:
                            all_data.append(formatted_data)
                    
                    except json.JSONDecodeError:
                        continue
        
        # Save unified training data
        output_path = self.output_dir / "unified_training_data.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for data in all_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"Created unified training data with {len(all_data)} samples")
        return str(output_path)
    
    def collect_all_datasets(self) -> Dict[str, str]:
        """Collect all available datasets"""
        logger.info("Collecting all financial datasets")
        
        all_files = {}
        
        # Collect FinanceBench
        financebench_files = self.process_financebench()
        all_files.update(financebench_files)
        
        # Collect XBRL Analysis
        xbrl_files = self.process_xbrl_analysis()
        all_files.update(xbrl_files)
        
        # Collect sentiment datasets
        sentiment_files = self.process_sentiment_datasets()
        all_files.update(sentiment_files)
        
        # Create unified training data
        unified_file = self.create_training_data(all_files)
        all_files['unified'] = unified_file
        
        return all_files

def main():
    """Example usage"""
    logging.basicConfig(level=logging.INFO)
    
    collector = HFDataCollector()
    
    # Collect all datasets
    collected_files = collector.collect_all_datasets()
    
    print("Collected datasets:")
    for name, path in collected_files.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()