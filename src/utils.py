"""
Utility functions for the OCR QA segmentation system.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_training_data(data_path: str) -> Tuple[List[List[Dict]], List[List[str]]]:
    """
    Load training data from JSON file.
    
    Expected format:
    [
        {
            "features": [...],  # List of feature dicts
            "labels": [...]     # List of tags
        },
        ...
    ]
    
    Args:
        data_path: Path to training data JSON
        
    Returns:
        (X, y) tuple of features and labels
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    X = [sample['features'] for sample in data]
    y = [sample['labels'] for sample in data]
    
    return X, y


def save_training_data(X: List[List[Dict]], y: List[List[str]], 
                      output_path: str):
    """
    Save training data to JSON file.
    
    Args:
        X: Features
        y: Labels
        output_path: Output file path
    """
    data = [
        {'features': features, 'labels': labels}
        for features, labels in zip(X, y)
    ]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Training data saved to: {output_file}")


def visualize_predictions(image_path: str, lines, tags, 
                         output_path: Optional[str] = None):
    """
    Visualize predictions by drawing bounding boxes on image.
    
    Args:
        image_path: Path to original image
        lines: OCR lines
        tags: Predicted tags
        output_path: Output image path (optional)
    """
    import cv2
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Color map for tags
    colors = {
        'B-Q': (0, 0, 255),    # Red
        'I-Q': (0, 100, 255),  # Orange
        'B-A': (0, 255, 0),    # Green
        'I-A': (100, 255, 100),# Light green
        'O': (128, 128, 128)   # Gray
    }
    
    # Draw bounding boxes
    for line, tag in zip(lines, tags):
        x, y, w, h = line.bbox
        color = colors.get(tag, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw tag label
        cv2.putText(img, tag, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Display or save
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), img)
        print(f"Visualization saved to: {output_file}")
    else:
        cv2.imshow('Predictions', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def calculate_iou(box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: (x, y, width, height)
        box2: (x, y, width, height)
        
    Returns:
        IoU score
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def print_qa_pairs(pairs):
    """
    Print question-answer pairs in a readable format.
    
    Args:
        pairs: List of QuestionAnswerPair objects
    """
    from postprocessing import QAPairExtractor

    
    extractor = QAPairExtractor()
    print(extractor.pairs_to_formatted_text(pairs))


def create_synthetic_training_data(n_samples: int = 10) -> Tuple[List[List[Dict]], List[List[str]]]:
    """
    Create synthetic training data for initial testing.
    
    Args:
        n_samples: Number of synthetic samples to create
        
    Returns:
        (X, y) tuple
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Create a simple pattern: Question -> Answer
        features = []
        labels = []
        
        # Question line
        features.append({
            'indent_level': '0.05',
            'vertical_gap': '2.0',
            'starts_with_q_marker': 'True',
            'starts_with_number': 'True',
            'word_count': '8',
            'prev_tag': 'O'
        })
        labels.append('B-Q')
        
        # Question continuation
        features.append({
            'indent_level': '0.05',
            'vertical_gap': '0.5',
            'starts_with_q_marker': 'False',
            'starts_with_number': 'False',
            'word_count': '10',
            'prev_tag': 'B-Q'
        })
        labels.append('I-Q')
        
        # Answer start
        features.append({
            'indent_level': '0.15',
            'vertical_gap': '1.5',
            'starts_with_q_marker': 'False',
            'starts_with_number': 'False',
            'word_count': '15',
            'prev_tag': 'I-Q'
        })
        labels.append('B-A')
        
        # Answer continuation
        for i in range(2):
            features.append({
                'indent_level': '0.15',
                'vertical_gap': '0.5',
                'starts_with_q_marker': 'False',
                'starts_with_number': 'False',
                'word_count': '12',
                'prev_tag': 'I-A' if i > 0 else 'B-A'
            })
            labels.append('I-A')
        
        X.append(features)
        y.append(labels)
    
    return X, y
