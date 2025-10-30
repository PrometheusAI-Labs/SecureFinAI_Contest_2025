#!/usr/bin/env python3
"""
FinGPT Compliance Agents - Main Pipeline Runner
Orchestrates the complete training and evaluation pipeline
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import subprocess
import time

# Add src to path
sys.path.append('src')

from data_collection.hf_data_collector import HFDataCollector
from data_collection.sec_data_collector import SECDataCollector
from data_collection.audio_data_collector import AudioDataCollector
from data_processing.data_processor import DataProcessor
from training.trainer import FinGPTTrainer
from evaluation.evaluator import FinGPTEvaluator
from testing.audio_tester import AudioTester
from submission.prepare_submission import SubmissionPreparer

logger = logging.getLogger(__name__)

class PipelineRunner:
    """Main pipeline runner for FinGPT Compliance Agents"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path(self.config['output']['results_path'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        self.pipeline_status = {
            'start_time': time.time(),
            'steps_completed': [],
            'steps_failed': [],
            'current_step': None
        }
    
    def run_step(self, step_name: str, step_func, *args, **kwargs):
        """Run a pipeline step with error handling"""
        logger.info(f"Starting step: {step_name}")
        self.pipeline_status['current_step'] = step_name
        
        try:
            result = step_func(*args, **kwargs)
            self.pipeline_status['steps_completed'].append(step_name)
            logger.info(f"Completed step: {step_name}")
            return result
        except Exception as e:
            logger.error(f"Failed step {step_name}: {e}")
            self.pipeline_status['steps_failed'].append({
                'step': step_name,
                'error': str(e)
            })
            return None
    
    def step_data_collection(self):
        """Step 1: Collect all required data"""
        logger.info("=== STEP 1: DATA COLLECTION ===")
        
        # Collect Hugging Face datasets
        hf_collector = HFDataCollector()
        hf_files = self.run_step(
            "hf_data_collection",
            hf_collector.collect_all_datasets
        )
        
        # Collect SEC data (limited for demo)
        sec_collector = SECDataCollector()
        sec_data = self.run_step(
            "sec_data_collection",
            sec_collector.collect_company_data,
            "Apple Inc.", "AAPL", 2  # Limited to 2 filings for demo
        )
        
        # Collect audio data
        audio_collector = AudioDataCollector()
        audio_data = self.run_step(
            "audio_data_collection",
            audio_collector.collect_all_audio_data
        )
        
        return {
            'hf_files': hf_files,
            'sec_data': sec_data,
            'audio_data': audio_data
        }
    
    def step_data_processing(self):
        """Step 2: Process and prepare data"""
        logger.info("=== STEP 2: DATA PROCESSING ===")
        
        processor = DataProcessor()
        self.run_step(
            "data_processing",
            processor.process_all_data
        )
        
        return True
    
    def step_model_training(self, mode: str = "lora"):
        """Step 3: Train the model"""
        logger.info("=== STEP 3: MODEL TRAINING ===")
        
        trainer = FinGPTTrainer()
        self.run_step(
            "model_training",
            trainer.train,
            mode
        )
        
        return True
    
    def step_model_evaluation(self):
        """Step 4: Evaluate the model"""
        logger.info("=== STEP 4: MODEL EVALUATION ===")
        
        evaluator = FinGPTEvaluator()
        results = self.run_step(
            "model_evaluation",
            evaluator.run_all_evaluations
        )
        
        return results
    
    def step_audio_testing(self):
        """Step 5: Test audio processing capabilities"""
        logger.info("=== STEP 5: AUDIO TESTING ===")
        
        tester = AudioTester()
        results = self.run_step(
            "audio_testing",
            tester.run_comprehensive_test
        )
        
        return results
    
    def step_prepare_submission(self):
        """Step 6: Prepare for submission"""
        logger.info("=== STEP 6: SUBMISSION PREPARATION ===")
        
        preparer = SubmissionPreparer()
        self.run_step(
            "submission_preparation",
            preparer.prepare_submission
        )
        
        return True
    
    def run_full_pipeline(self, mode: str = "lora"):
        """Run the complete pipeline"""
        logger.info("Starting FinGPT Compliance Agents Pipeline")
        logger.info(f"Training mode: {mode}")
        
        start_time = time.time()
        
        # Step 1: Data Collection
        data_results = self.step_data_collection()
        
        # Step 2: Data Processing
        self.step_data_processing()
        
        # Step 3: Model Training
        self.step_model_training(mode)
        
        # Step 4: Model Evaluation
        eval_results = self.step_model_evaluation()
        
        # Step 5: Audio Testing
        audio_results = self.step_audio_testing()
        
        # Step 6: Submission Preparation
        self.step_prepare_submission()
        
        # Calculate total time
        total_time = time.time() - start_time
        self.pipeline_status['total_time'] = total_time
        self.pipeline_status['end_time'] = time.time()
        
        # Save pipeline status
        status_file = self.results_dir / 'pipeline_status.json'
        with open(status_file, 'w') as f:
            json.dump(self.pipeline_status, f, indent=2)
        
        # Print summary
        self.print_pipeline_summary()
        
        return {
            'data_results': data_results,
            'eval_results': eval_results,
            'audio_results': audio_results,
            'pipeline_status': self.pipeline_status
        }
    
    def run_quick_test(self):
        """Run a quick test pipeline with minimal data"""
        logger.info("Running quick test pipeline")
        
        # Only collect HF data and train with LoRA
        hf_collector = HFDataCollector()
        hf_files = self.run_step(
            "quick_hf_collection",
            hf_collector.collect_all_datasets
        )
        
        processor = DataProcessor()
        self.run_step(
            "quick_data_processing",
            processor.process_all_data
        )
        
        trainer = FinGPTTrainer()
        self.run_step(
            "quick_training",
            trainer.train,
            "lora"
        )
        
        evaluator = FinGPTEvaluator()
        results = self.run_step(
            "quick_evaluation",
            evaluator.run_all_evaluations
        )
        
        return results
    
    def print_pipeline_summary(self):
        """Print pipeline execution summary"""
        print("\n" + "="*60)
        print("FINGPT COMPLIANCE AGENTS - PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Total execution time: {self.pipeline_status['total_time']:.2f} seconds")
        print(f"Steps completed: {len(self.pipeline_status['steps_completed'])}")
        print(f"Steps failed: {len(self.pipeline_status['steps_failed'])}")
        
        print("\nCompleted steps:")
        for step in self.pipeline_status['steps_completed']:
            print(f"  ✓ {step}")
        
        if self.pipeline_status['steps_failed']:
            print("\nFailed steps:")
            for failure in self.pipeline_status['steps_failed']:
                print(f"  ✗ {failure['step']}: {failure['error']}")
        
        print("\nResults saved to:")
        print(f"  - Pipeline status: {self.results_dir / 'pipeline_status.json'}")
        print(f"  - Evaluation results: {self.results_dir / 'evaluation_results.json'}")
        print(f"  - Audio test results: {self.results_dir / 'audio_test_results.json'}")
        print(f"  - Submission package: submission/")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Run FinGPT Compliance Agents Pipeline')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['lora', 'full'], default='lora', help='Training mode')
    parser.add_argument('--quick', action='store_true', help='Run quick test pipeline')
    
    args = parser.parse_args()
    
    runner = PipelineRunner(args.config)
    
    if args.quick:
        results = runner.run_quick_test()
    else:
        results = runner.run_full_pipeline(args.mode)
    
    print("\nPipeline execution completed!")

if __name__ == "__main__":
    main()