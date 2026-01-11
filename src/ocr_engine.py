"""
OCR engine module with support for PaddleOCR and Tesseract.

Extracts text lines with bounding boxes and confidence scores.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class OCRLine:
    """Represents a single line of text extracted by OCR."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    line_number: int


class OCREngine:
    """OCR engine wrapper supporting multiple backends."""
    
    def __init__(self, engine: str = 'paddleocr', lang: str = 'eng'):
        """
        Initialize OCR engine.
        
        Args:
            engine: OCR backend ('paddleocr' or 'tesseract')
            lang: Language code
        """
        self.engine_name = engine.lower()
        self.lang = lang
        self.ocr = None
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine_name == 'paddleocr':
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    show_log=False,
                    use_gpu=False
                )
            except ImportError:
                raise ImportError(
                    "PaddleOCR not installed. Install with: pip install paddleocr"
                )
        elif self.engine_name == 'tesseract':
            try:
                import pytesseract
                self.ocr = pytesseract
            except ImportError:
                raise ImportError(
                    "pytesseract not installed. Install with: pip install pytesseract"
                )
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine_name}")
    
    def _bbox_to_xyxy(self, bbox_points: List) -> Tuple[int, int, int, int]:
        """
        Convert bounding box points to (x, y, width, height) format.
        
        Args:
            bbox_points: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            (x, y, width, height) tuple
        """
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        
        x = int(min(xs))
        y = int(min(ys))
        width = int(max(xs) - x)
        height = int(max(ys) - y)
        
        return (x, y, width, height)
    
    def extract_lines_paddleocr(self, image: np.ndarray) -> List[OCRLine]:
        """
        Extract text lines using PaddleOCR.
        
        Args:
            image: Input image
            
        Returns:
            List of OCRLine objects
        """
        result = self.ocr.ocr(image, cls=True)
        
        if result is None or len(result) == 0 or result[0] is None:
            return []
        
        lines = []
        for idx, line_data in enumerate(result[0]):
            bbox_points = line_data[0]
            text_info = line_data[1]
            text = text_info[0]
            confidence = float(text_info[1])
            
            bbox = self._bbox_to_xyxy(bbox_points)
            
            lines.append(OCRLine(
                text=text,
                bbox=bbox,
                confidence=confidence,
                line_number=idx
            ))
        
        return lines
    
    def extract_lines_tesseract(self, image: np.ndarray) -> List[OCRLine]:
        """
        Extract text lines using Tesseract.
        
        Args:
            image: Input image
            
        Returns:
            List of OCRLine objects
        """
        import pytesseract
        from PIL import Image
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
        
        # Get detailed data
        data = pytesseract.image_to_data(pil_image, lang=self.lang, 
                                        output_type=pytesseract.Output.DICT)
        
        # Group by line
        lines_dict = {}
        for i in range(len(data['text'])):
            if data['text'][i].strip() == '':
                continue
            
            block_num = data['block_num'][i]
            par_num = data['par_num'][i]
            line_num = data['line_num'][i]
            
            key = (block_num, par_num, line_num)
            
            if key not in lines_dict:
                lines_dict[key] = {
                    'text': [],
                    'left': [],
                    'top': [],
                    'width': [],
                    'height': [],
                    'conf': []
                }
            
            lines_dict[key]['text'].append(data['text'][i])
            lines_dict[key]['left'].append(data['left'][i])
            lines_dict[key]['top'].append(data['top'][i])
            lines_dict[key]['width'].append(data['width'][i])
            lines_dict[key]['height'].append(data['height'][i])
            lines_dict[key]['conf'].append(data['conf'][i])
        
        # Convert to OCRLine objects
        lines = []
        for idx, (key, line_data) in enumerate(sorted(lines_dict.items())):
            text = ' '.join(line_data['text'])
            
            # Calculate bounding box for entire line
            x = min(line_data['left'])
            y = min(line_data['top'])
            width = max([l + w for l, w in zip(line_data['left'], line_data['width'])]) - x
            height = max([t + h for t, h in zip(line_data['top'], line_data['height'])]) - y
            
            # Average confidence
            valid_confs = [c for c in line_data['conf'] if c != -1]
            confidence = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
            confidence = confidence / 100.0  # Normalize to 0-1
            
            lines.append(OCRLine(
                text=text,
                bbox=(x, y, width, height),
                confidence=confidence,
                line_number=idx
            ))
        
        return lines
    
    def extract_lines(self, image: np.ndarray) -> List[OCRLine]:
        """
        Extract text lines from image.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            List of OCRLine objects sorted by vertical position
        """
        if self.engine_name == 'paddleocr':
            lines = self.extract_lines_paddleocr(image)
        elif self.engine_name == 'tesseract':
            lines = self.extract_lines_tesseract(image)
        else:
            raise ValueError(f"Unsupported engine: {self.engine_name}")
        
        # Sort lines by vertical position (top to bottom)
        lines.sort(key=lambda l: l.bbox[1])
        
        # Update line numbers after sorting
        for idx, line in enumerate(lines):
            line.line_number = idx
        
        return lines
    
    def get_image_dimensions(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image: Input image
            
        Returns:
            (width, height) tuple
        """
        h, w = image.shape[:2]
        return (w, h)
