#!/usr/bin/env python3
"""
FinGPT Compliance Agents - Quick Start Script
Demonstrates the complete pipeline with minimal setup
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'transformers', 'datasets', 'peft', 
        'pandas', 'numpy', 'requests', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: uv sync")
        return False
    
    print("✅ All dependencies available!")
    return True

def create_sample_data():
    """Create sample data for demonstration"""
    print("\n📊 Creating sample data...")
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample financial Q&A data
    sample_qa_data = [
        {
            "question": "What is the company's revenue for Q3 2024?",
            "answer": "The company reported revenue of $50.2 million for Q3 2024, representing a 15% increase year-over-year.",
            "context": "Based on the quarterly earnings report..."
        },
        {
            "question": "What are the main risks mentioned in the 10-K filing?",
            "answer": "The main risks include market volatility, regulatory changes, and competitive pressures in the technology sector.",
            "context": "From the risk factors section of the 10-K filing..."
        }
    ]
    
    # Save sample data
    with open(data_dir / "sample_qa.jsonl", 'w') as f:
        for item in sample_qa_data:
            f.write(json.dumps(item) + '\n')
    
    # Sample sentiment data
    sample_sentiment_data = [
        {
            "text": "The company reported strong quarterly earnings growth.",
            "gold": 2  # positive
        },
        {
            "text": "Market volatility continues to impact investor confidence.",
            "gold": 0  # negative
        },
        {
            "text": "The Federal Reserve maintained interest rates at current levels.",
            "gold": 1  # neutral
        }
    ]
    
    with open(data_dir / "sample_sentiment.jsonl", 'w') as f:
        for item in sample_sentiment_data:
            f.write(json.dumps(item) + '\n')
    
    print("✅ Sample data created!")

def demonstrate_data_collection():
    """Demonstrate data collection capabilities"""
    print("\n🔍 Demonstrating data collection...")
    
    try:
        from data_collection.hf_data_collector import HFDataCollector
        
        collector = HFDataCollector()
        print("✅ HF Data Collector initialized")
        
        # Note: In a real scenario, this would collect actual data
        print("📝 Note: Full data collection requires API keys and network access")
        print("   - Hugging Face datasets: FinanceBench, XBRL Analysis, Sentiment")
        print("   - SEC EDGAR: Company filings and XBRL data")
        print("   - Financial APIs: Real-time market data")
        print("   - Audio data: Earnings calls, podcasts, news")
        
    except Exception as e:
        print(f"❌ Error in data collection demo: {e}")

def demonstrate_data_processing():
    """Demonstrate data processing capabilities"""
    print("\n⚙️ Demonstrating data processing...")
    
    try:
        from data_processing.data_processor import DataProcessor
        
        processor = DataProcessor()
        print("✅ Data Processor initialized")
        
        # Process sample data
        print("📝 Processing sample data...")
        # Note: This would process the actual data in a real scenario
        
    except Exception as e:
        print(f"❌ Error in data processing demo: {e}")

def demonstrate_model_training():
    """Demonstrate model training capabilities"""
    print("\n🤖 Demonstrating model training...")
    
    try:
        from training.trainer import FinGPTTrainer
        
        trainer = FinGPTTrainer()
        print("✅ FinGPT Trainer initialized")
        
        print("📝 Training configuration:")
        print("   - Base model: meta-llama/Llama-3.1-8B-Instruct")
        print("   - Fine-tuning: LoRA (Low-Rank Adaptation)")
        print("   - Tasks: Financial Q&A, Sentiment Analysis, XBRL Processing")
        print("   - Max parameters: 8B (contest requirement)")
        
    except Exception as e:
        print(f"❌ Error in training demo: {e}")

def demonstrate_evaluation():
    """Demonstrate evaluation capabilities"""
    print("\n📊 Demonstrating evaluation...")
    
    try:
        from evaluation.evaluator import FinGPTEvaluator
        
        evaluator = FinGPTEvaluator()
        print("✅ FinGPT Evaluator initialized")
        
        print("📝 Evaluation tasks:")
        print("   - Financial Q&A: Accuracy on FinanceBench")
        print("   - Sentiment Analysis: F1-score on financial sentiment")
        print("   - XBRL Processing: Tag/value extraction accuracy")
        print("   - Audio Processing: Word Error Rate (WER)")
        
    except Exception as e:
        print(f"❌ Error in evaluation demo: {e}")

def demonstrate_audio_processing():
    """Demonstrate audio processing capabilities"""
    print("\n🎵 Demonstrating audio processing...")
    
    try:
        from testing.audio_tester import AudioTester
        
        tester = AudioTester()
        print("✅ Audio Tester initialized")
        
        print("📝 Audio processing capabilities:")
        print("   - Speech recognition for financial content")
        print("   - Sentiment analysis on audio transcripts")
        print("   - Topic classification from audio")
        print("   - Integration with OpenAI Whisper/GPT-4o")
        
    except Exception as e:
        print(f"❌ Error in audio processing demo: {e}")

def demonstrate_submission():
    """Demonstrate submission preparation"""
    print("\n📦 Demonstrating submission preparation...")
    
    try:
        from submission.prepare_submission import SubmissionPreparer
        
        preparer = SubmissionPreparer()
        print("✅ Submission Preparer initialized")
        
        print("📝 Submission package includes:")
        print("   - Model weights and configuration")
        print("   - Inference scripts for all tasks")
        print("   - Requirements and documentation")
        print("   - Evaluation scripts")
        print("   - Hugging Face model card")
        
    except Exception as e:
        print(f"❌ Error in submission demo: {e}")

def show_next_steps():
    """Show next steps for the user"""
    print("\n🚀 Next Steps:")
    print("=============")
    print("1. Configure your environment:")
    print("   cp .env.example .env")
    print("   # Edit .env with your API keys")
    print()
    print("2. Run the complete pipeline:")
    print("   make all")
    print("   # or")
    print("   uv run python run_pipeline.py --mode lora")
    print()
    print("3. Run a quick test:")
    print("   make quick-test")
    print("   # or")
    print("   uv run python run_pipeline.py --quick")
    print()
    print("4. Check project status:")
    print("   make status")
    print()
    print("5. View logs:")
    print("   make logs")

def main():
    """Main demonstration function"""
    print("🎯 FinGPT Compliance Agents - Quick Start Demo")
    print("=" * 50)
    
    setup_logging()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create sample data
    create_sample_data()
    
    # Demonstrate each component
    demonstrate_data_collection()
    demonstrate_data_processing()
    demonstrate_model_training()
    demonstrate_evaluation()
    demonstrate_audio_processing()
    demonstrate_submission()
    
    # Show next steps
    show_next_steps()
    
    print("\n✅ Quick start demo completed!")
    print("The project is ready for development and training.")

if __name__ == "__main__":
    main()