#!/usr/bin/env python3
"""
Inference Script for SecureFinAI Contest 2025 Task 3
OCR Task: Convert financial document images to structured HTML

This script provides a simple interface for inference on single images or batches.
"""

import argparse
import base64
import sys
from pathlib import Path
from improved_ocr_agent import improved_agent_from_image


def inference_from_base64(b64_image: str) -> str:
    """
    Run inference on a base64-encoded image.
    
    Args:
        b64_image: Base64-encoded image string
        
    Returns:
        Structured HTML string
    """
    return improved_agent_from_image(b64_image)


def inference_from_file(image_path: str) -> str:
    """
    Run inference on an image file.
    
    Args:
        image_path: Path to image file (PNG, JPG, etc.)
        
    Returns:
        Structured HTML string
    """
    # Read image and convert to base64
    with open(image_path, 'rb') as f:
        img_data = f.read()
        b64_image = base64.b64encode(img_data).decode('utf-8')
    
    return improved_agent_from_image(b64_image)


def main():
    parser = argparse.ArgumentParser(
        description='OCR Inference for Financial Documents - SecureFinAI Contest 2025 Task 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From base64 string (stdin)
  echo "iVBORw0KGgo..." | python inference.py --base64
  
  # From image file
  python inference.py --image document.png
  
  # Save output to file
  python inference.py --image document.png --output result.html
        """
    )
    
    parser.add_argument(
        '--base64',
        action='store_true',
        help='Read base64 image from stdin'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output HTML file path (default: stdout)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.base64 and not args.image:
        parser.error("Either --base64 or --image must be specified")
    
    if args.base64 and args.image:
        parser.error("Cannot specify both --base64 and --image")
    
    try:
        # Run inference
        if args.base64:
            # Read from stdin
            b64_image = sys.stdin.read().strip()
            html_output = inference_from_base64(b64_image)
        else:
            # Read from file
            if not Path(args.image).exists():
                print(f"Error: Image file not found: {args.image}", file=sys.stderr)
                sys.exit(1)
            html_output = inference_from_file(args.image)
        
        # Output result
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"✅ Output saved to: {args.output}")
        else:
            print(html_output)
            
    except Exception as e:
        print(f"❌ Error during inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

