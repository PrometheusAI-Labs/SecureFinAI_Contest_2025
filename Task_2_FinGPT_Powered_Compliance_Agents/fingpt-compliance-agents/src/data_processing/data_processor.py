#!/usr/bin/env python3
"""
Data Processor for FinGPT Compliance Agents
Processes and prepares collected data for training
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataProcessor:
    """Processes and prepares data for training"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = Path(self.config['output']['data_path'])
        self.processed_dir = self.data_dir / 'processed'
        self.train_dir = self.data_dir / 'train'
        self.test_dir = self.data_dir / 'test'
        
        # Create directories
        for dir_path in [self.processed_dir, self.train_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_financial_qa_data(self, input_file: str) -> List[Dict]:
        """Process financial Q&A data"""
        logger.info(f"Processing financial Q&A data from {input_file}")
        
        processed_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Create training example
                    example = {
                        'instruction': 'Answer the following financial question based on the provided context.',
                        'input': f"Context: {data.get('context', '')}\nQuestion: {data.get('question', '')}",
                        'output': data.get('answer', ''),
                        'task': 'financial_qa',
                        'source': 'financebench'
                    }
                    
                    processed_data.append(example)
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Processed {len(processed_data)} financial Q&A examples")
        return processed_data
    
    def process_sentiment_data(self, input_file: str) -> List[Dict]:
        """Process sentiment analysis data"""
        logger.info(f"Processing sentiment data from {input_file}")
        
        processed_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Map sentiment labels
                    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                    sentiment = label_map.get(data.get('gold', 1), 'neutral')
                    
                    # Create training example
                    example = {
                        'instruction': 'Analyze the sentiment of the following financial text.',
                        'input': f"Text: {data.get('text', '')}",
                        'output': f"Sentiment: {sentiment}",
                        'task': 'sentiment_analysis',
                        'source': 'sentiment_dataset'
                    }
                    
                    processed_data.append(example)
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Processed {len(processed_data)} sentiment examples")
        return processed_data
    
    def process_xbrl_data(self, input_file: str) -> List[Dict]:
        """Process XBRL analysis data"""
        logger.info(f"Processing XBRL data from {input_file}")
        
        processed_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Create training example for XBRL tasks
                    if 'input' in data and 'output' in data:
                        example = {
                            'instruction': 'Extract the requested information from the XBRL data.',
                            'input': f"XBRL Data: {data.get('input', '')}",
                            'output': data.get('output', ''),
                            'task': 'xbrl_analysis',
                            'source': 'xbrl_dataset'
                        }
                        
                        processed_data.append(example)
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Processed {len(processed_data)} XBRL examples")
        return processed_data
    
    def process_xbrl_training_data(self, input_file: str) -> List[Dict]:
        """Process XBRL training data from collected specifications"""
        logger.info(f"Processing XBRL training data from {input_file}")
        
        processed_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Create training example for XBRL tasks
                    if 'instruction' in data and 'input' in data and 'output' in data:
                        example = {
                            'instruction': data.get('instruction', ''),
                            'input': data.get('input', ''),
                            'output': data.get('output', ''),
                            'task': data.get('task', 'xbrl_extraction'),
                            'source': data.get('source', 'xbrl_specifications')
                        }
                        
                        processed_data.append(example)
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Processed {len(processed_data)} XBRL training examples")
        return processed_data
    
    def process_sec_filings_data(self, input_file: str) -> List[Dict]:
        """Process SEC filings data"""
        logger.info(f"Processing SEC filings data from {input_file}")
        
        processed_data = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                sec_data = json.load(f)
            
            # Process XBRL data from SEC filings
            for filing_data in sec_data.get('xbrl_data', []):
                xbrl_data = filing_data.get('data', {})
                
                # Create examples for different XBRL tasks
                for concept, details in xbrl_data.items():
                    if isinstance(details, dict) and 'value' in details:
                        example = {
                            'instruction': 'Extract financial information from SEC filing data.',
                            'input': f"Concept: {concept}\nContext: {details.get('context_ref', '')}",
                            'output': f"Value: {details['value']}",
                            'task': 'sec_filing_analysis',
                            'source': 'sec_edgar'
                        }
                        
                        processed_data.append(example)
        
        except Exception as e:
            logger.error(f"Error processing SEC filings data: {e}")
        
        logger.info(f"Processed {len(processed_data)} SEC filing examples")
        return processed_data
    
    def create_training_format(self, processed_data: List[Dict]) -> List[Dict]:
        """Convert processed data to training format"""
        logger.info("Converting to training format")
        
        training_data = []
        
        for example in processed_data:
            # Create conversation format for training
            conversation = [
                {
                    "role": "system",
                    "content": "You are a financial compliance expert AI assistant specialized in SEC filings analysis, regulatory compliance, sentiment analysis, and financial data processing."
                },
                {
                    "role": "user", 
                    "content": f"{example['instruction']}\n\n{example['input']}"
                },
                {
                    "role": "assistant",
                    "content": example['output']
                }
            ]
            
            training_example = {
                'conversations': conversation,
                'task': example['task'],
                'source': example['source']
            }
            
            training_data.append(training_example)
        
        logger.info(f"Created {len(training_data)} training examples")
        return training_data
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.8) -> tuple:
        """Split data into train and test sets"""
        logger.info(f"Splitting data with train ratio: {train_ratio}")
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        # Split data
        split_idx = int(len(data) * train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        logger.info(f"Split: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data
    
    def save_data(self, data: List[Dict], filepath: str):
        """Save data to JSONL file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} examples to {filepath}")
    
    def process_all_data(self):
        """Process all collected data"""
        logger.info("Starting data processing pipeline")
        
        all_processed_data = []
        
        # Process different data sources
        data_sources = [
            ('financebench', self.process_financial_qa_data),
            ('sentiment', self.process_sentiment_data),
            ('xbrl', self.process_xbrl_data),
            ('xbrl_training', self.process_xbrl_training_data),
            ('sec_filings', self.process_sec_filings_data)
        ]
        
        for source_name, processor_func in data_sources:
            # Look for input files
            input_files = list(self.data_dir.glob(f"raw/**/*{source_name}*.jsonl"))
            
            for input_file in input_files:
                try:
                    processed = processor_func(str(input_file))
                    all_processed_data.extend(processed)
                except Exception as e:
                    logger.error(f"Error processing {input_file}: {e}")
        
        if not all_processed_data:
            logger.warning("No data processed. Check input files.")
            return
        
        # Convert to training format
        training_data = self.create_training_format(all_processed_data)
        
        # Split data
        train_data, test_data = self.split_data(training_data)
        
        # Save processed data
        self.save_data(train_data, self.train_dir / "train.jsonl")
        self.save_data(test_data, self.test_dir / "test.jsonl")
        
        # Save statistics
        stats = {
            'total_examples': len(training_data),
            'train_examples': len(train_data),
            'test_examples': len(test_data),
            'tasks': list(set(ex['task'] for ex in training_data)),
            'sources': list(set(ex['source'] for ex in training_data))
        }
        
        with open(self.processed_dir / "data_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Data processing completed!")
        logger.info(f"Statistics: {stats}")

def main():
    parser = argparse.ArgumentParser(description='Process data for FinGPT Compliance Agents')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.config)
    processor.process_all_data()

if __name__ == "__main__":
    main()