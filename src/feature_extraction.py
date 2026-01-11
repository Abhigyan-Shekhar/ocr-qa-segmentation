"""
Feature extraction module for CRF-based sequence labeling.

Extracts visual and textual features from OCR lines without using LLMs.
"""

import re
import numpy as np
from typing import List, Dict, Optional
from ocr_engine import OCRLine
import Levenshtein



class FeatureExtractor:
    """Extract features from OCR lines for CRF model."""
    
    def __init__(self, image_width: int, image_height: int):
        """
        Initialize feature extractor.
        
        Args:
            image_width: Width of the processed image
            image_height: Height of the processed image
        """
        self.image_width = image_width
        self.image_height = image_height
        
        # Patterns for text-based features
        self.question_patterns = [
            r'^[Qq](?:uestion)?[\s\.]?\d+',  # Q1, Question 1, Q.1
            r'^\d+[\s\.\):]',                 # 1., 1), 1:
            r'^[Qq]\d+',                      # Q1, q1
        ]
        
        self.number_pattern = r'^\d+'
        self.punct_pattern = r'[?.!:;]$'
    
    def _normalize_position(self, value: float, max_value: float) -> float:
        """Normalize position to [0, 1] range."""
        return value / max_value if max_value > 0 else 0.0
    
    def _calculate_indentation(self, line: OCRLine) -> float:
        """
        Calculate normalized indentation level.
        
        Args:
            line: OCR line
            
        Returns:
            Indentation ratio (0-1)
        """
        x = line.bbox[0]
        return self._normalize_position(x, self.image_width)
    
    def _calculate_vertical_gap(self, current_line: OCRLine, 
                               prev_line: Optional[OCRLine]) -> float:
        """
        Calculate vertical gap from previous line.
        
        Args:
            current_line: Current OCR line
            prev_line: Previous OCR line (None if first line)
            
        Returns:
            Normalized vertical gap
        """
        if prev_line is None:
            return 0.0
        
        current_y = current_line.bbox[1]
        prev_y = prev_line.bbox[1]
        prev_height = prev_line.bbox[3]
        
        gap = current_y - (prev_y + prev_height)
        
        # Normalize by average line height (approximate)
        avg_line_height = 30.0  # Typical line height
        return max(0.0, gap / avg_line_height)
    
    def _calculate_line_length(self, line: OCRLine) -> float:
        """
        Calculate normalized line length.
        
        Args:
            line: OCR line
            
        Returns:
            Length ratio (0-1)
        """
        width = line.bbox[2]
        return self._normalize_position(width, self.image_width)
    
    def _starts_with_question_marker(self, text: str) -> bool:
        """Check if text starts with question marker."""
        text = text.strip()
        for pattern in self.question_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _fuzzy_starts_with_q(self, text: str) -> bool:
        """
        Fuzzy match for 'Q' at start (handles OCR errors).
        
        Args:
            text: Input text
            
        Returns:
            True if likely starts with Q
        """
        if len(text) < 1:
            return False
        
        first_char = text[0].upper()
        
        # Direct match
        if first_char == 'Q':
            return True
        
        # Common OCR errors: O, 0, D
        if first_char in ['O', '0', 'D']:
            # Check if followed by number
            if len(text) > 1 and text[1].isdigit():
                return True
        
        return False
    
    def _starts_with_number(self, text: str) -> bool:
        """Check if text starts with a number."""
        text = text.strip()
        return bool(re.match(self.number_pattern, text))
    
    def _ends_with_punctuation(self, text: str) -> bool:
        """Check if text ends with question-like punctuation."""
        text = text.strip()
        return bool(re.search(self.punct_pattern, text))
    
    def _calculate_word_count(self, text: str) -> int:
        """Calculate number of words."""
        return len(text.split())
    
    def _calculate_avg_char_spacing(self, line: OCRLine) -> float:
        """
        Calculate average character spacing.
        
        Args:
            line: OCR line
            
        Returns:
            Average spacing (normalized)
        """
        text_length = len(line.text)
        if text_length == 0:
            return 0.0
        
        bbox_width = line.bbox[2]
        return bbox_width / text_length / 20.0  # Normalize by typical char width
    
    def _is_capitalized(self, text: str) -> bool:
        """Check if first word is capitalized."""
        text = text.strip()
        if len(text) == 0:
            return False
        
        words = text.split()
        if len(words) == 0:
            return False
        
        return words[0][0].isupper()
    
    def _is_all_caps(self, text: str) -> bool:
        """Check if text is mostly uppercase."""
        text = text.strip()
        if len(text) == 0:
            return False
        
        letters = [c for c in text if c.isalpha()]
        if len(letters) == 0:
            return False
        
        upper_count = sum(1 for c in letters if c.isupper())
        return upper_count / len(letters) > 0.7
    
    def _calculate_horizontal_alignment(self, line: OCRLine) -> str:
        """
        Determine horizontal alignment (left, center, right).
        
        Args:
            line: OCR line
            
        Returns:
            Alignment category
        """
        x = line.bbox[0]
        width = line.bbox[2]
        center_x = x + width / 2
        
        page_center = self.image_width / 2
        left_margin = self.image_width * 0.3
        right_margin = self.image_width * 0.7
        
        if center_x < left_margin:
            return 'left'
        elif center_x > right_margin:
            return 'right'
        else:
            return 'center'
    
    def extract_features(self, lines: List[OCRLine], 
                        prev_tags: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract features for all lines.
        
        Args:
            lines: List of OCR lines
            prev_tags: Previous predictions (for inference with context)
            
        Returns:
            List of feature dictionaries
        """
        features = []
        
        for idx, line in enumerate(lines):
            prev_line = lines[idx - 1] if idx > 0 else None
            prev_tag = prev_tags[idx - 1] if prev_tags and idx > 0 else 'O'
            
            feat = self._extract_single_line_features(line, prev_line, prev_tag)
            features.append(feat)
        
        return features
    
    def _extract_single_line_features(self, line: OCRLine, 
                                     prev_line: Optional[OCRLine],
                                     prev_tag: str) -> Dict:
        """
        Extract features for a single line.
        
        Args:
            line: Current OCR line
            prev_line: Previous OCR line
            prev_tag: Previous line's tag
            
        Returns:
            Feature dictionary
        """
        text = line.text.strip()
        
        # Visual features
        indent = self._calculate_indentation(line)
        vgap = self._calculate_vertical_gap(line, prev_line)
        line_length = self._calculate_line_length(line)
        alignment = self._calculate_horizontal_alignment(line)
        avg_spacing = self._calculate_avg_char_spacing(line)
        
        # Text features
        starts_q = self._starts_with_question_marker(text)
        fuzzy_q = self._fuzzy_starts_with_q(text)
        starts_num = self._starts_with_number(text)
        ends_punct = self._ends_with_punctuation(text)
        word_count = self._calculate_word_count(text)
        is_cap = self._is_capitalized(text)
        is_all_caps = self._is_all_caps(text)
        
        # OCR quality
        confidence = line.confidence
        
        # Position features
        is_first_line = (line.line_number == 0)
        
        # Build feature dictionary
        features = {
            # Visual features
            'indent_level': round(indent, 3),
            'vertical_gap': round(vgap, 2),
            'line_length': round(line_length, 3),
            'alignment': alignment,
            'avg_char_spacing': round(avg_spacing, 2),
            
            # Text pattern features
            'starts_with_q_marker': starts_q,
            'fuzzy_starts_q': fuzzy_q,
            'starts_with_number': starts_num,
            'ends_with_punct': ends_punct,
            'is_capitalized': is_cap,
            'is_all_caps': is_all_caps,
            
            # Content features
            'word_count': word_count,
            'word_count_bin': 'short' if word_count < 3 else 'medium' if word_count < 10 else 'long',
            'text_length': len(text),
            
            # OCR quality
            'ocr_confidence': round(confidence, 2),
            'low_confidence': confidence < 0.5,
            
            # Position
            'is_first_line': is_first_line,
            
            # Context (from previous line)
            'prev_tag': prev_tag,
            
            # Combined features
            'q_marker_and_short': starts_q and word_count < 15,
            'large_gap': vgap > 2.0,
            'high_indent': indent > 0.1,
        }
        
        return features
    
    def features_to_crf_format(self, features: List[Dict]) -> List[Dict]:
        """
        Convert features to CRF-compatible format (all string values).
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            CRF-formatted feature list
        """
        crf_features = []
        
        for feat in features:
            crf_feat = {}
            for key, value in feat.items():
                if isinstance(value, bool):
                    crf_feat[key] = str(value)
                elif isinstance(value, (int, float)):
                    crf_feat[key] = str(value)
                else:
                    crf_feat[key] = str(value)
            crf_features.append(crf_feat)
        
        return crf_features
