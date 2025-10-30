#!/usr/bin/env python3
"""
Basic tests for FinGPT Compliance Agents
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from data_collection.hf_data_collector import HFDataCollector
        from data_collection.sec_data_collector import SECDataCollector
        from data_collection.audio_data_collector import AudioDataCollector
        from data_processing.data_processor import DataProcessor
        from training.trainer import FinGPTTrainer
        from evaluation.evaluator import FinGPTEvaluator
        from testing.audio_tester import AudioTester
        from submission.prepare_submission import SubmissionPreparer
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")

def test_config_loading():
    """Test that configuration can be loaded"""
    try:
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml.example"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        assert 'model' in config
        assert 'lora' in config
        assert 'training' in config
    except Exception as e:
        pytest.fail(f"Failed to load configuration: {e}")

def test_data_collectors():
    """Test data collector initialization"""
    try:
        from data_collection.hf_data_collector import HFDataCollector
        from data_collection.sec_data_collector import SECDataCollector
        from data_collection.audio_data_collector import AudioDataCollector
        
        # Test HF collector
        hf_collector = HFDataCollector()
        assert hf_collector is not None
        
        # Test SEC collector
        sec_collector = SECDataCollector()
        assert sec_collector is not None
        
        # Test Audio collector
        audio_collector = AudioDataCollector()
        assert audio_collector is not None
        
    except Exception as e:
        pytest.fail(f"Failed to initialize data collectors: {e}")

def test_data_processor():
    """Test data processor initialization"""
    try:
        from data_processing.data_processor import DataProcessor
        processor = DataProcessor()
        assert processor is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize data processor: {e}")

def test_trainer():
    """Test trainer initialization"""
    try:
        from training.trainer import FinGPTTrainer
        trainer = FinGPTTrainer()
        assert trainer is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize trainer: {e}")

def test_evaluator():
    """Test evaluator initialization"""
    try:
        from evaluation.evaluator import FinGPTEvaluator
        evaluator = FinGPTEvaluator()
        assert evaluator is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize evaluator: {e}")

def test_audio_tester():
    """Test audio tester initialization"""
    try:
        from testing.audio_tester import AudioTester
        tester = AudioTester()
        assert tester is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize audio tester: {e}")

def test_submission_preparer():
    """Test submission preparer initialization"""
    try:
        from submission.prepare_submission import SubmissionPreparer
        preparer = SubmissionPreparer()
        assert preparer is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize submission preparer: {e}")

if __name__ == "__main__":
    pytest.main([__file__])