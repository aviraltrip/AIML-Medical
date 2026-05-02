import os
from pathlib import Path
from typing import Optional

import pdfplumber
import pytesseract
from PIL import Image, ImageOps

class OCREngine:
    """PulsePoint OCR Engine for digital and scanned medical reports."""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Optimizes image for OCR: Grayscale, Invert if needed, and Rescale."""
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        # Auto-level contrast
        image = ImageOps.autocontrast(image)
        return image

    def extract_from_pdf(self, pdf_path: Path) -> str:
        """Extracts text directly from digital PDF pages."""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
        return text

    def extract_from_image(self, image_path: Path) -> str:
        """Extracts text from images (JPG, PNG) using Tesseract OCR."""
        try:
            with Image.open(image_path) as img:
                processed_img = self.preprocess_image(img)
                # config='--psm 6' assumes a single uniform block of text
                text = pytesseract.image_to_string(processed_img, config='--psm 3')
                return text
        except Exception as e:
            return f"Error reading image: {str(e)}"

    def process_file(self, file_path: str) -> str:
        """Routes file to the appropriate extractor based on extension."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        if ext == ".pdf":
            return self.extract_from_pdf(path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            return self.extract_from_image(path)
        else:
            return f"Unsupported file type: {ext}"

# Global instance for easy import
ocr_engine = OCREngine()
