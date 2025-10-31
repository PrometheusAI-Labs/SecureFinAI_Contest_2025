"""
Improved OCR Agent for Financial Document Processing
SecureFinAI Contest 2025 Task 3

This module provides an enhanced OCR agent that:
1. Uses advanced OCR techniques with multiple engines
2. Implements financial document structure recognition
3. Generates structured HTML with proper formatting
4. Handles tables, headers, and financial data formatting
"""

import base64
import io
import re
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


@dataclass
class OCRResult:
    """Container for OCR results with confidence scores"""
    text: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    engine: str = ""


class FinancialDocumentProcessor:
    """Enhanced processor for financial documents with structure recognition"""
    
    def __init__(self, use_multiple_engines: bool = True):
        self.use_multiple_engines = use_multiple_engines
        self.ocr_engines = self._initialize_ocr_engines()
        
    def _initialize_ocr_engines(self) -> List[str]:
        """Initialize available OCR engines"""
        engines = []
        
        if TESSERACT_AVAILABLE:
            engines.append("tesseract")
        if EASYOCR_AVAILABLE:
            engines.append("easyocr")
        if PADDLEOCR_AVAILABLE:
            engines.append("paddleocr")
            
        return engines
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing for better OCR results"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Convert to numpy for OpenCV processing
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.medianBlur(thresh, 3)
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(denoised)
        
        return processed_image
    
    def extract_text_tesseract(self, image: Image.Image) -> OCRResult:
        """Extract text using Tesseract with financial document optimization"""
        try:
            # Use financial document specific config
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()[]{}:;!@#$%^&*()_+-=<>?/|\\"\'`~ '
            
            text = pytesseract.image_to_string(
                image, 
                lang='eng',
                config=custom_config
            )
            
            # Get confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,
                engine="tesseract"
            )
        except Exception as e:
            return OCRResult(text="", confidence=0.0, engine="tesseract")
    
    def extract_text_easyocr(self, image: Image.Image) -> OCRResult:
        """Extract text using EasyOCR"""
        try:
            reader = easyocr.Reader(['en'])
            img_array = np.array(image)
            results = reader.readtext(img_array)
            
            text_parts = []
            total_confidence = 0
            
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    text_parts.append(text)
                    total_confidence += confidence
            
            combined_text = ' '.join(text_parts)
            avg_confidence = total_confidence / len(text_parts) if text_parts else 0
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine="easyocr"
            )
        except Exception as e:
            return OCRResult(text="", confidence=0.0, engine="easyocr")
    
    def extract_text_paddleocr(self, image: Image.Image) -> OCRResult:
        """Extract text using PaddleOCR"""
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            img_array = np.array(image)
            results = ocr.ocr(img_array, cls=True)
            
            text_parts = []
            total_confidence = 0
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.5:
                            text_parts.append(text)
                            total_confidence += confidence
            
            combined_text = ' '.join(text_parts)
            avg_confidence = total_confidence / len(text_parts) if text_parts else 0
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine="paddleocr"
            )
        except Exception as e:
            return OCRResult(text="", confidence=0.0, engine="paddleocr")
    
    def extract_text_multiple_engines(self, image: Image.Image) -> OCRResult:
        """Extract text using multiple OCR engines and combine results"""
        results = []
        
        if "tesseract" in self.ocr_engines:
            results.append(self.extract_text_tesseract(image))
        
        if "easyocr" in self.ocr_engines:
            results.append(self.extract_text_easyocr(image))
            
        if "paddleocr" in self.ocr_engines:
            results.append(self.extract_text_paddleocr(image))
        
        # Filter out empty results
        valid_results = [r for r in results if r.text.strip()]
        
        if not valid_results:
            return OCRResult(text="", confidence=0.0, engine="combined")
        
        # Use the result with highest confidence
        best_result = max(valid_results, key=lambda x: x.confidence)
        
        return best_result
    
    def detect_financial_structure(self, text: str) -> Dict[str, List[str]]:
        """Detect financial document structure elements"""
        structure = {
            'headers': [],
            'tables': [],
            'numbers': [],
            'dates': [],
            'sections': []
        }
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect headers (all caps, short lines, or lines ending with colon)
            if (line.isupper() and len(line) < 100) or line.endswith(':'):
                structure['headers'].append(line)
            
            # Detect numbers (financial amounts, percentages)
            number_pattern = r'\$?[\d,]+\.?\d*%?'
            numbers = re.findall(number_pattern, line)
            if numbers:
                structure['numbers'].extend(numbers)
            
            # Detect dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\w+\s+\d{1,2},?\s+\d{4}\b'
            dates = re.findall(date_pattern, line)
            if dates:
                structure['dates'].extend(dates)
            
            # Detect table-like content (lines with multiple numbers or separators)
            if re.search(r'[\|\t]', line) or len(re.findall(number_pattern, line)) > 2:
                structure['tables'].append(line)
        
        return structure
    
    def generate_structured_html(self, text: str, structure: Dict[str, List[str]]) -> str:
        """Generate structured HTML from OCR text and detected structure"""
        lines = text.split('\n')
        html_parts = ['<html><body>']
        
        current_section = None
        in_table = False
        table_rows = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a header
            is_header = (line in structure['headers'] or 
                        line.isupper() and len(line) < 100 or 
                        line.endswith(':'))
            
            # Check if this is table content
            is_table_content = (line in structure['tables'] or 
                               re.search(r'[\|\t]', line) or 
                               len(re.findall(r'\$?[\d,]+\.?\d*%?', line)) > 2)
            
            if is_header:
                # Close previous table if open
                if in_table and table_rows:
                    html_parts.append(self._format_table(table_rows))
                    table_rows = []
                    in_table = False
                
                # Add header with appropriate level
                if line.isupper() and len(line) < 50:
                    html_parts.append(f'<h1 class="main-header">{line}</h1>')
                else:
                    html_parts.append(f'<h2 class="section-header">{line}</h2>')
                current_section = line
                
            elif is_table_content:
                # Start or continue table
                if not in_table:
                    in_table = True
                    table_rows = []
                
                # Parse table row
                cells = re.split(r'[\|\t]+', line)
                cells = [cell.strip() for cell in cells if cell.strip()]
                if cells:
                    table_rows.append(cells)
                    
            else:
                # Close table if open
                if in_table and table_rows:
                    html_parts.append(self._format_table(table_rows))
                    table_rows = []
                    in_table = False
                
                # Add regular paragraph
                if line:
                    # Check if line contains financial numbers
                    if re.search(r'\$?[\d,]+\.?\d*%?', line):
                        html_parts.append(f'<p class="financial-data">{line}</p>')
                    elif re.search(r'\b(Total|Revenue|Income|Assets|Liabilities|Equity)\b', line, re.IGNORECASE):
                        html_parts.append(f'<p class="financial-summary">{line}</p>')
                    else:
                        html_parts.append(f'<p>{line}</p>')
        
        # Close any remaining table
        if in_table and table_rows:
            html_parts.append(self._format_table(table_rows))
        
        html_parts.append('</body></html>')
        
        return '\n'.join(html_parts)
    
    def _format_table(self, rows: List[List[str]]) -> str:
        """Format table rows as HTML table"""
        if not rows:
            return ""
        
        html = ['<table class="financial-table">']
        
        for i, row in enumerate(rows):
            if i == 0:
                # First row as header
                html.append('<thead><tr>')
                for cell in row:
                    html.append(f'<th>{cell}</th>')
                html.append('</tr></thead><tbody>')
            else:
                html.append('<tr>')
                for cell in row:
                    # Check if cell contains numbers
                    if re.search(r'\$?[\d,]+\.?\d*%?', cell):
                        html.append(f'<td class="number">{cell}</td>')
                    else:
                        html.append(f'<td>{cell}</td>')
                html.append('</tr>')
        
        html.append('</tbody></table>')
        return '\n'.join(html)
    
    def process_document(self, b64_img: str) -> str:
        """Main method to process a financial document from base64 image"""
        if not isinstance(b64_img, str) or not b64_img.strip():
            return "<html><body><p></p></body></html>"
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(b64_img)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Extract text using multiple engines
            if self.use_multiple_engines and len(self.ocr_engines) > 1:
                ocr_result = self.extract_text_multiple_engines(processed_image)
            else:
                # Use tesseract as fallback
                ocr_result = self.extract_text_tesseract(processed_image)
            
            # Detect document structure
            structure = self.detect_financial_structure(ocr_result.text)
            
            # Generate structured HTML
            html = self.generate_structured_html(ocr_result.text, structure)
            
            return html
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return "<html><body><p>Error processing document</p></body></html>"


def improved_agent_from_image(b64_img: str) -> str:
    """Improved agent function that processes base64 image to structured HTML"""
    processor = FinancialDocumentProcessor(use_multiple_engines=True)
    return processor.process_document(b64_img)


def baseline_agent_from_image(b64_img: str) -> str:
    """Original baseline agent for comparison"""
    if not isinstance(b64_img, str) or not b64_img.strip():
        return "<html><body><p></p></body></html>"
    try:
        img_data = base64.b64decode(b64_img)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        text = ""
    return f"<html><body><p>{text.strip()}</p></body></html>"