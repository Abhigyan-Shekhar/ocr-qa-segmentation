"""
OCR and Question-Answer Segmentation System

A classical CV/ML system for digitizing handwritten exams and separating
questions from answers without using LLMs.
"""

__version__ = "1.0.0"
__author__ = "Abhigyan Shekhar"

from .preprocessing import ImagePreprocessor
from .ocr_engine import OCREngine
from .feature_extraction import FeatureExtractor
from .crf_model import CRFModel
from .postprocessing import QAPairExtractor

__all__ = [
    "ImagePreprocessor",
    "OCREngine",
    "FeatureExtractor",
    "CRFModel",
    "QAPairExtractor",
]
