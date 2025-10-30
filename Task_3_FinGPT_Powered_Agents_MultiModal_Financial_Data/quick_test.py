#!/usr/bin/env python3
"""
Quick test script for the improved OCR agent
"""

import base64
from improved_ocr_agent import improved_agent_from_image, baseline_agent_from_image

def test_with_sample_image():
    """Test with a simple base64 encoded image"""
    # 1x1 pixel PNG image (base64 encoded)
    sample_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    print("🧪 Testing improved OCR agent...")
    
    # Test improved agent
    improved_result = improved_agent_from_image(sample_b64)
    print(f"✅ Improved agent result: {improved_result}")
    
    # Test baseline agent
    baseline_result = baseline_agent_from_image(sample_b64)
    print(f"📊 Baseline agent result: {baseline_result}")
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_with_sample_image()