#!/usr/bin/env python3
"""
Main script for SecureFinAI Contest 2025 Task 3
FinGPT-Powered Agents for MultiModal Financial Data

This script provides an improved OCR solution with:
- Multiple OCR engines (Tesseract, EasyOCR, PaddleOCR)
- Advanced image preprocessing
- Financial document structure recognition
- Enhanced HTML generation with proper formatting
- Comprehensive evaluation metrics
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset
from improved_ocr_agent import improved_agent_from_image, baseline_agent_from_image
from improved_evaluation import EnhancedEvaluator


def main():
    parser = argparse.ArgumentParser(description='SecureFinAI Contest 2025 Task 3 - Improved OCR Solution')
    parser.add_argument('--dataset', default='TheFinAI/SecureFinAI_Contest_2025-Task_3_EnglishOCR',
                       help='HuggingFace dataset repository')
    parser.add_argument('--max-samples', type=int, default=10,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--output-dir', default='./results',
                       help='Output directory for predictions and results')
    parser.add_argument('--model-name', default='improved',
                       help='Model name for output files')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare with baseline agent')
    parser.add_argument('--lang', default='en', choices=['en', 'es'],
                       help='Language (en for English, es for Spanish)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Starting SecureFinAI Contest 2025 Task 3 - Improved OCR Solution")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🌍 Language: {args.lang}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔢 Max samples: {args.max_samples}")
    print("="*60)
    
    # Load dataset
    print("📥 Loading dataset...")
    try:
        dataset = load_dataset(args.dataset, split="test")
        data = dataset.to_pandas()
        print(f"✅ Loaded dataset with {len(data)} samples")
    except Exception as e:
        print(f"❌ Could not load {args.dataset}: {e}")
        print("📝 Creating toy dataset for testing...")
        data = pd.DataFrame({
            "image": [
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",  # 1x1 pixel
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="  # 1x1 pixel
            ],
            "matched_html": [
                "<html><body><h2>Financial Report</h2><p>Revenue: $1.2M</p><table><tr><th>Q1</th><th>Q2</th></tr><tr><td>$300K</td><td>$400K</td></tr></table></body></html>",
                "<html><body><h2>Balance Sheet</h2><p>Assets: $5.5M</p><p>Liabilities: $2.1M</p></body></html>"
            ]
        })
        print("✅ Created toy dataset for testing")
    
    # Limit samples if specified
    if args.max_samples is not None and len(data) > args.max_samples:
        data = data.head(args.max_samples)
        print(f"📊 Using {len(data)} samples for evaluation")
    
    # Initialize evaluator
    print("🔧 Initializing evaluator...")
    evaluator = EnhancedEvaluator()
    
    # Process samples
    print("⚙️ Processing samples...")
    results = []
    improved_predictions = []
    baseline_predictions = []
    ground_truths = []
    
    for i, row in data.iterrows():
        print(f"  Processing sample {i+1}/{len(data)}...")
        
        b64_img = row.get("image", "")
        gt_html = row.get("matched_html", "")
        
        # Generate predictions
        improved_pred = improved_agent_from_image(b64_img)
        improved_predictions.append(improved_pred)
        ground_truths.append(gt_html)
        
        if args.compare_baseline:
            baseline_pred = baseline_agent_from_image(b64_img)
            baseline_predictions.append(baseline_pred)
        
        # Save prediction to file
        pred_file = os.path.join(args.output_dir, f"{args.model_name}_pred_{i}.html")
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write(improved_pred)
    
    print(f"✅ Generated {len(improved_predictions)} predictions")
    
    # Run evaluation
    print("📊 Running evaluation...")
    eval_results = evaluator.comprehensive_evaluation(improved_predictions, ground_truths)
    
    # Print results
    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS")
    print("="*60)
    print(f"ROUGE-1: {eval_results['rouge1']:.4f}")
    print(f"ROUGE-2: {eval_results['rouge2']:.4f}")
    print(f"ROUGE-L: {eval_results['rougeL']:.4f}")
    print(f"BLEU: {eval_results['bleu']:.4f}")
    
    if 'tag_jaccard' in eval_results:
        print(f"HTML Tag Jaccard: {eval_results['tag_jaccard']:.4f}")
    
    if 'number_f1' in eval_results:
        print(f"Financial Numbers F1: {eval_results['number_f1']:.4f}")
    
    if 'date_f1' in eval_results:
        print(f"Date F1: {eval_results['date_f1']:.4f}")
    
    # Compare with baseline if requested
    if args.compare_baseline and baseline_predictions:
        print("\n" + "="*60)
        print("🔄 BASELINE COMPARISON")
        print("="*60)
        
        baseline_eval = evaluator.comprehensive_evaluation(baseline_predictions, ground_truths)
        
        print(f"Improved ROUGE-1: {eval_results['rouge1']:.4f}")
        print(f"Baseline ROUGE-1: {baseline_eval['rouge1']:.4f}")
        improvement = eval_results['rouge1'] - baseline_eval['rouge1']
        print(f"Improvement: {improvement:+.4f} ({improvement/baseline_eval['rouge1']*100:+.1f}%)")
        
        if 'tag_jaccard' in eval_results and 'tag_jaccard' in baseline_eval:
            print(f"Improved HTML Structure: {eval_results['tag_jaccard']:.4f}")
            print(f"Baseline HTML Structure: {baseline_eval['tag_jaccard']:.4f}")
            struct_improvement = eval_results['tag_jaccard'] - baseline_eval['tag_jaccard']
            print(f"Structure Improvement: {struct_improvement:+.4f}")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "evaluation_results.csv")
    detailed_results = []
    
    for i, (pred, gt) in enumerate(zip(improved_predictions, ground_truths)):
        sample_rouge = evaluator.rouge.compute(predictions=[pred], references=[gt], use_stemmer=True)
        sample_structure = evaluator.evaluate_html_structure(pred, gt)
        sample_financial = evaluator.evaluate_financial_content(pred, gt)
        
        detailed_results.append({
            'index': i,
            'rouge1': sample_rouge['rouge1'],
            'rouge2': sample_rouge['rouge2'],
            'rougeL': sample_rouge['rougeL'],
            'tag_jaccard': sample_structure.get('tag_jaccard', 0),
            'number_f1': sample_financial.get('number_f1', 0),
            'prediction_length': len(pred),
            'reference_length': len(gt)
        })
    
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(results_file, index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"  📁 Predictions: {args.output_dir}/")
    print(f"  📊 Evaluation: {results_file}")
    
    print("\n🎉 Evaluation complete!")
    print("\n📋 Next steps for further improvement:")
    print("1. Fine-tune with FinGPT on financial document datasets")
    print("2. Implement vision-language models (CLIP, BLIP)")
    print("3. Add domain-specific financial knowledge")
    print("4. Implement advanced table detection")
    print("5. Add multi-language support")


if __name__ == "__main__":
    main()