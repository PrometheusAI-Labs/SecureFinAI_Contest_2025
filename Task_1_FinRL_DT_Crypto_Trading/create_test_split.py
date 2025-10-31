#!/usr/bin/env python3
"""
Create Test Data Split for Crypto Trading Evaluation

This script splits the training data into train/test sets and creates
the BTC_1sec_with_sentiment_risk_test.csv file for evaluation.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse

def create_test_split(input_file, output_dir, test_size=0.2, random_state=42):
    """
    Split the training data into train/test sets
    
    Args:
        input_file: Path to the training CSV file
        output_dir: Directory to save the split files
        test_size: Fraction of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility
    """
    print(f"Loading data from: {input_file}")
    
    # Load the training data
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the data
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Save the split data
    train_file = os.path.join(output_dir, "BTC_1sec_with_sentiment_risk_train.csv")
    test_file = os.path.join(output_dir, "BTC_1sec_with_sentiment_risk_test.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"✅ Train data saved to: {train_file}")
    print(f"✅ Test data saved to: {test_file}")
    
    # Print some statistics
    print("\nData Split Statistics:")
    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    print(f"  Train percentage: {len(train_df)/len(df)*100:.1f}%")
    print(f"  Test percentage: {len(test_df)/len(df)*100:.1f}%")
    
    return train_file, test_file

def main():
    parser = argparse.ArgumentParser(description="Create test data split for crypto trading evaluation")
    parser.add_argument("--input_file", 
                       default="offline_data_preparation/data/BTC_1sec_with_sentiment_risk_train.csv",
                       help="Path to input training data")
    parser.add_argument("--output_dir", 
                       default="offline_data_preparation/data",
                       help="Directory to save split files")
    parser.add_argument("--test_size", 
                       type=float, 
                       default=0.2,
                       help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--random_state", 
                       type=int, 
                       default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CRYPTO DATA SPLIT CREATION")
    print("=" * 60)
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return 1
    
    try:
        train_file, test_file = create_test_split(
            args.input_file, 
            args.output_dir, 
            args.test_size, 
            args.random_state
        )
        
        print("\n✅ Data split completed successfully!")
        print(f"   Train file: {train_file}")
        print(f"   Test file: {test_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during data split: {e}")
        return 1

if __name__ == "__main__":
    exit(main())