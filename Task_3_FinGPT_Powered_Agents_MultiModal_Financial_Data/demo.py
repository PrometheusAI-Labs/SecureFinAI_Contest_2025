#!/usr/bin/env python3
"""
Demo script for SecureFinAI Contest 2025 Task 3
Shows the capabilities of the improved OCR agent
"""

import base64
import os
from improved_ocr_agent import improved_agent_from_image, baseline_agent_from_image, FinancialDocumentProcessor

def create_sample_financial_document():
    """Create a sample financial document for demonstration"""
    # This would normally be a real financial document image
    # For demo purposes, we'll use a simple base64 encoded image
    sample_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    return sample_b64

def demo_ocr_capabilities():
    """Demonstrate OCR capabilities"""
    print("🎯 SecureFinAI Contest 2025 Task 3 - OCR Agent Demo")
    print("="*60)
    
    # Initialize processor
    processor = FinancialDocumentProcessor(use_multiple_engines=True)
    print(f"🔧 Available OCR engines: {processor.ocr_engines}")
    
    # Create sample document
    sample_doc = create_sample_financial_document()
    
    print("\n📄 Sample Financial Document Processing:")
    print("-" * 40)
    
    # Process with improved agent
    improved_result = improved_agent_from_image(sample_doc)
    print("✅ Improved Agent Result:")
    print(improved_result)
    
    # Process with baseline agent
    baseline_result = baseline_agent_from_image(sample_doc)
    print("\n📊 Baseline Agent Result:")
    print(baseline_result)
    
    print("\n🔍 Key Improvements:")
    print("• Multiple OCR engines (Tesseract, EasyOCR, PaddleOCR)")
    print("• Advanced image preprocessing")
    print("• Financial document structure recognition")
    print("• Enhanced HTML generation with proper formatting")
    print("• Comprehensive evaluation metrics")

def demo_structure_recognition():
    """Demonstrate structure recognition capabilities"""
    print("\n🏗️ Financial Document Structure Recognition:")
    print("-" * 50)
    
    # Sample financial text
    sample_text = """
    FINANCIAL REPORT Q3 2024
    
    Revenue: $1,250,000
    Operating Income: $350,000
    Net Profit: $280,000
    
    BALANCE SHEET
    Assets: $5,500,000
    Liabilities: $2,100,000
    Equity: $3,400,000
    """
    
    processor = FinancialDocumentProcessor()
    structure = processor.detect_financial_structure(sample_text)
    
    print("Detected Structure:")
    for key, values in structure.items():
        if values:
            print(f"  {key}: {values}")

def demo_html_generation():
    """Demonstrate HTML generation capabilities"""
    print("\n🌐 Enhanced HTML Generation:")
    print("-" * 40)
    
    sample_text = """
    QUARTERLY EARNINGS REPORT
    
    Revenue: $1,250,000
    Operating Income: $350,000
    Net Profit: $280,000
    
    BALANCE SHEET
    Assets: $5,500,000
    Liabilities: $2,100,000
    Equity: $3,400,000
    """
    
    processor = FinancialDocumentProcessor()
    structure = processor.detect_financial_structure(sample_text)
    html = processor.generate_structured_html(sample_text, structure)
    
    print("Generated HTML:")
    print(html)

def main():
    """Main demo function"""
    demo_ocr_capabilities()
    demo_structure_recognition()
    demo_html_generation()
    
    print("\n" + "="*60)
    print("🚀 Ready for SecureFinAI Contest 2025 Task 3!")
    print("Run 'python main.py --help' for evaluation options")
    print("="*60)

if __name__ == "__main__":
    main()