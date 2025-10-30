#!/usr/bin/env python3
"""
Data Download Script for FinRL Crypto Trading Project
Downloads required datasets for the SecureFinAI Contest 2025 Task 1

Datasets:
1. FinRL_BTC_news_signals from Hugging Face
2. BTC_1sec_with_sentiment_risk_train.csv from Google Drive (backup/update)
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dependencies():
    """Install required dependencies"""
    try:
        import huggingface_hub
        import gdown
        import datasets
        logger.info("All required dependencies are already installed")
        return True
    except ImportError as e:
        logger.info(f"Installing missing dependencies: {e}")
        os.system("pip install huggingface_hub gdown datasets")
        return True

def download_huggingface_dataset():
    """Download FinRL_BTC_news_signals dataset from Hugging Face"""
    try:
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset
        
        logger.info("Downloading FinRL_BTC_news_signals from Hugging Face...")
        
        # Create data directory if it doesn't exist
        data_dir = Path("offline_data_preparation/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the dataset
        dataset = load_dataset("SecureFinAI-Lab/FinRL_BTC_news_signals")
        
        # Save as CSV for easier access
        if 'train' in dataset:
            train_df = dataset['train'].to_pandas()
            output_path = data_dir / "FinRL_BTC_news_signals.csv"
            train_df.to_csv(output_path, index=False)
            logger.info(f"Saved FinRL_BTC_news_signals to {output_path}")
            logger.info(f"Dataset shape: {train_df.shape}")
            logger.info(f"Columns: {list(train_df.columns)}")
            return True
        else:
            logger.error("No 'train' split found in the dataset")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading Hugging Face dataset: {e}")
        return False

def download_google_drive_dataset():
    """Download BTC_1sec_with_sentiment_risk_train.csv from Google Drive"""
    try:
        import gdown
        
        logger.info("Downloading BTC_1sec_with_sentiment_risk_train.csv from Google Drive...")
        
        # Google Drive file ID (extracted from the URL)
        file_id = "1rV9tJ0T2iWNJ-g3TI4Qgqy0cVf_Zqzqp"
        
        # Create data directory if it doesn't exist
        data_dir = Path("offline_data_preparation/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        output_path = data_dir / "BTC_1sec_with_sentiment_risk_train_gdrive.csv"
        gdown.download_folder(f"https://drive.google.com/drive/folders/{file_id}", 
                            output=str(data_dir), 
                            quiet=False, 
                            use_cookies=False)
        
        # Check if the file was downloaded
        if (data_dir / "BTC_1sec_with_sentiment_risk_train.csv").exists():
            logger.info(f"Successfully downloaded BTC_1sec_with_sentiment_risk_train.csv")
            return True
        else:
            logger.warning("File not found after download, trying alternative method...")
            # Try direct file download
            gdown.download(f"https://drive.google.com/uc?id={file_id}", 
                         str(output_path), 
                         quiet=False)
            return output_path.exists()
            
    except Exception as e:
        logger.error(f"Error downloading Google Drive dataset: {e}")
        return False

def verify_datasets():
    """Verify downloaded datasets"""
    data_dir = Path("offline_data_preparation/data")
    
    logger.info("Verifying downloaded datasets...")
    
    # Check FinRL_BTC_news_signals
    hf_file = data_dir / "FinRL_BTC_news_signals.csv"
    if hf_file.exists():
        df = pd.read_csv(hf_file)
        logger.info(f"FinRL_BTC_news_signals: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"First few rows:\n{df.head()}")
    else:
        logger.warning("FinRL_BTC_news_signals.csv not found")
    
    # Check BTC_1sec_with_sentiment_risk_train
    btc_file = data_dir / "BTC_1sec_with_sentiment_risk_train.csv"
    if btc_file.exists():
        df = pd.read_csv(btc_file, nrows=5)  # Read only first 5 rows for verification
        logger.info(f"BTC_1sec_with_sentiment_risk_train: {df.shape[1]} columns")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"First few rows:\n{df.head()}")
    else:
        logger.warning("BTC_1sec_with_sentiment_risk_train.csv not found")

def main():
    """Main function to download all required datasets"""
    logger.info("Starting data download process...")
    
    # Setup dependencies
    if not setup_dependencies():
        logger.error("Failed to setup dependencies")
        return False
    
    # Download datasets
    hf_success = download_huggingface_dataset()
    gdrive_success = download_google_drive_dataset()
    
    # Verify datasets
    verify_datasets()
    
    if hf_success and gdrive_success:
        logger.info("All datasets downloaded successfully!")
        return True
    else:
        logger.warning("Some datasets may not have been downloaded successfully")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)