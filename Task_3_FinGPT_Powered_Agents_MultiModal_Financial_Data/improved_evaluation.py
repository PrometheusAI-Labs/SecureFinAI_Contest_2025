"""
Improved Evaluation Script for OCR Task
SecureFinAI Contest 2025 Task 3

This script provides enhanced evaluation capabilities including:
1. Multiple OCR engine comparison
2. Detailed performance metrics
3. HTML structure analysis
4. Financial document specific evaluation
"""

import os
import base64
import io
import re
import math
import pandas as pd
from tqdm import tqdm
from PIL import Image
import evaluate
from improved_ocr_agent import improved_agent_from_image, baseline_agent_from_image, FinancialDocumentProcessor
from datasets import load_dataset


class EnhancedEvaluator:
    """Enhanced evaluator with multiple metrics and analysis capabilities"""
    
    def __init__(self):
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        
    def evaluate_html_structure(self, predicted_html: str, ground_truth_html: str) -> dict:
        """Evaluate HTML structure similarity"""
        metrics = {}
        
        # Extract HTML tags
        pred_tags = re.findall(r'<[^>]+>', predicted_html)
        gt_tags = re.findall(r'<[^>]+>', ground_truth_html)
        
        # Count different types of tags
        pred_tag_counts = {}
        gt_tag_counts = {}
        
        for tag in pred_tags:
            tag_name = re.sub(r'[<>/]', '', tag).split()[0]
            pred_tag_counts[tag_name] = pred_tag_counts.get(tag_name, 0) + 1
            
        for tag in gt_tags:
            tag_name = re.sub(r'[<>/]', '', tag).split()[0]
            gt_tag_counts[tag_name] = gt_tag_counts.get(tag_name, 0) + 1
        
        # Calculate tag overlap
        all_tags = set(pred_tag_counts.keys()) | set(gt_tag_counts.keys())
        if all_tags:
            tag_overlap = sum(min(pred_tag_counts.get(tag, 0), gt_tag_counts.get(tag, 0)) 
                            for tag in all_tags)
            tag_union = sum(max(pred_tag_counts.get(tag, 0), gt_tag_counts.get(tag, 0)) 
                          for tag in all_tags)
            metrics['tag_jaccard'] = tag_overlap / tag_union if tag_union > 0 else 0
        else:
            metrics['tag_jaccard'] = 0
        
        # Count structural elements
        structural_elements = ['table', 'tr', 'td', 'th', 'h1', 'h2', 'h3', 'p', 'div', 'span']
        for element in structural_elements:
            pred_count = pred_tag_counts.get(element, 0)
            gt_count = gt_tag_counts.get(element, 0)
            metrics[f'{element}_count_diff'] = abs(pred_count - gt_count)
        
        return metrics
    
    def evaluate_financial_content(self, predicted_html: str, ground_truth_html: str) -> dict:
        """Evaluate financial content specific metrics"""
        metrics = {}
        
        # Extract financial numbers
        financial_pattern = r'\$?[\d,]+\.?\d*%?'
        pred_numbers = re.findall(financial_pattern, predicted_html)
        gt_numbers = re.findall(financial_pattern, ground_truth_html)
        
        # Calculate number overlap
        if gt_numbers:
            pred_set = set(pred_numbers)
            gt_set = set(gt_numbers)
            intersection = pred_set & gt_set
            union = pred_set | gt_set
            
            metrics['number_precision'] = len(intersection) / len(pred_set) if pred_set else 0
            metrics['number_recall'] = len(intersection) / len(gt_set) if gt_set else 0
            metrics['number_f1'] = (2 * metrics['number_precision'] * metrics['number_recall'] / 
                                  (metrics['number_precision'] + metrics['number_recall'])) if (metrics['number_precision'] + metrics['number_recall']) > 0 else 0
        else:
            metrics['number_precision'] = 0
            metrics['number_recall'] = 0
            metrics['number_f1'] = 0
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b'
        pred_dates = re.findall(date_pattern, predicted_html)
        gt_dates = re.findall(date_pattern, ground_truth_html)
        
        if gt_dates:
            pred_date_set = set(pred_dates)
            gt_date_set = set(gt_dates)
            date_intersection = pred_date_set & gt_date_set
            
            metrics['date_precision'] = len(date_intersection) / len(pred_date_set) if pred_date_set else 0
            metrics['date_recall'] = len(date_intersection) / len(gt_date_set) if gt_date_set else 0
            metrics['date_f1'] = (2 * metrics['date_precision'] * metrics['date_recall'] / 
                                (metrics['date_precision'] + metrics['date_recall'])) if (metrics['date_precision'] + metrics['date_recall']) > 0 else 0
        else:
            metrics['date_precision'] = 0
            metrics['date_recall'] = 0
            metrics['date_f1'] = 0
        
        return metrics
    
    def comprehensive_evaluation(self, predictions: list, references: list) -> dict:
        """Perform comprehensive evaluation with multiple metrics"""
        results = {}
        
        # Basic ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        results.update({
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL']
        })
        
        # BLEU scores
        try:
            bleu_scores = self.bleu.compute(
                predictions=predictions,
                references=references
            )
            results['bleu'] = bleu_scores['bleu']
        except:
            results['bleu'] = 0.0
        
        # HTML structure evaluation
        structure_metrics = []
        financial_metrics = []
        
        for pred, ref in zip(predictions, references):
            structure_metrics.append(self.evaluate_html_structure(pred, ref))
            financial_metrics.append(self.evaluate_financial_content(pred, ref))
        
        # Average structure metrics
        if structure_metrics:
            structure_avg = {}
            for key in structure_metrics[0].keys():
                structure_avg[key] = sum(m[key] for m in structure_metrics) / len(structure_metrics)
            results.update(structure_avg)
        
        # Average financial metrics
        if financial_metrics:
            financial_avg = {}
            for key in financial_metrics[0].keys():
                financial_avg[key] = sum(m[key] for m in financial_metrics) / len(financial_metrics)
            results.update(financial_avg)
        
        return results


def run_improved_evaluation(hf_repo: str, pred_dir: str, model_name: str, 
                          lang: str, output_csv: str, max_samples: int = None):
    """Run improved evaluation with comprehensive metrics"""
    
    # Load dataset
    try:
        dataset = load_dataset(hf_repo, split="test")
        data = dataset.to_pandas()
    except:
        print(f"Could not load {hf_repo}, using toy data")
        data = pd.DataFrame({
            "image": [],
            "matched_html": [
                "<html><body><p>Total revenue for Q1 was $1.2B.</p></body></html>",
                "<html><body><p>Operating income increased by 12% year-over-year.</p></body></html>"
            ]
        })
    
    if max_samples is not None and len(data) > max_samples:
        data = data.head(max_samples)
    
    print(f"Evaluating {len(data)} samples...")
    
    # Initialize evaluator
    evaluator = EnhancedEvaluator()
    
    # Generate predictions
    predictions = []
    references = []
    
    for i in tqdm(data.index, desc="Generating predictions"):
        b64_img = data.loc[i, "image"] if "image" in data.columns else ""
        gt_html = data.loc[i, "matched_html"]
        
        # Generate prediction using improved agent
        pred_html = improved_agent_from_image(b64_img)
        
        predictions.append(pred_html)
        references.append(gt_html)
    
    # Comprehensive evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(predictions, references)
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print(f"BLEU: {results['bleu']:.4f}")
    
    if 'tag_jaccard' in results:
        print(f"HTML Tag Jaccard: {results['tag_jaccard']:.4f}")
    
    if 'number_f1' in results:
        print(f"Financial Numbers F1: {results['number_f1']:.4f}")
    
    if 'date_f1' in results:
        print(f"Date F1: {results['date_f1']:.4f}")
    
    # Save detailed results
    detailed_results = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        row = {
            'index': i,
            'rouge1': evaluator.rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)['rouge1'],
            'prediction_length': len(pred),
            'reference_length': len(ref)
        }
        
        # Add structure metrics
        structure = evaluator.evaluate_html_structure(pred, ref)
        row.update(structure)
        
        # Add financial metrics
        financial = evaluator.evaluate_financial_content(pred, ref)
        row.update(financial)
        
        detailed_results.append(row)
    
    df_results = pd.DataFrame(detailed_results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")
    
    return df_results, results


if __name__ == "__main__":
    # Configuration
    hf_repo = "TheFinAI/SecureFinAI_Contest_2025-Task_3_EnglishOCR"
    pred_dir = "./preds_improved"
    model_name = "improved"
    lang = "en"
    eval_output = "./eval_improved_comprehensive.csv"
    max_samples = 5  # Set to None for all samples
    
    # Run evaluation
    df_results, summary_results = run_improved_evaluation(
        hf_repo, pred_dir, model_name, lang, eval_output, max_samples
    )