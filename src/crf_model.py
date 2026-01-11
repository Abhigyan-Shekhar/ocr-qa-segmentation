"""
CRF model module for sequence labeling.

Uses Conditional Random Fields to classify text lines as questions or answers.
"""

import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sklearn_crfsuite
from sklearn_crfsuite import metrics


class CRFModel:
    """CRF model for question-answer segmentation."""
    
    # Label set: BIO tagging scheme
    LABELS = ['B-Q', 'I-Q', 'B-A', 'I-A', 'O']
    
    def __init__(self, 
                 algorithm: str = 'lbfgs',
                 c1: float = 0.1,
                 c2: float = 0.1,
                 max_iterations: int = 100,
                 all_possible_transitions: bool = True):
        """
        Initialize CRF model.
        
        Args:
            algorithm: Training algorithm
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: Maximum training iterations
            all_possible_transitions: Whether to include all possible label transitions
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=all_possible_transitions,
            verbose=False
        )
        
        self.is_trained = False
    
    def train(self, X_train: List[List[Dict]], y_train: List[List[str]],
             X_val: Optional[List[List[Dict]]] = None,
             y_val: Optional[List[List[str]]] = None) -> Dict:
        """
        Train the CRF model.
        
        Args:
            X_train: Training features (list of sequences)
            y_train: Training labels (list of label sequences)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Training metrics dictionary
        """
        print(f"Training CRF model on {len(X_train)} sequences...")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_f1 = metrics.flat_f1_score(y_train, train_pred, 
                                         average='weighted', 
                                         labels=self.LABELS)
        
        results = {
            'train_f1': train_f1,
            'train_samples': len(X_train)
        }
        
        print(f"Training F1: {train_f1:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_f1 = metrics.flat_f1_score(y_val, val_pred,
                                          average='weighted',
                                          labels=self.LABELS)
            
            results['val_f1'] = val_f1
            results['val_samples'] = len(X_val)
            
            print(f"Validation F1: {val_f1:.4f}")
            
            # Detailed validation metrics
            print("\nValidation Classification Report:")
            print(metrics.flat_classification_report(
                y_val, val_pred, labels=self.LABELS, digits=3
            ))
        
        return results
    
    def predict(self, X: List[List[Dict]]) -> List[List[str]]:
        """
        Predict labels for sequences.
        
        Args:
            X: Feature sequences
            
        Returns:
            Predicted label sequences
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_single(self, features: List[Dict]) -> List[str]:
        """
        Predict labels for a single sequence.
        
        Args:
            features: Feature list for one document
            
        Returns:
            Predicted label list
        """
        predictions = self.predict([features])
        return predictions[0]
    
    def predict_marginals(self, X: List[List[Dict]]) -> List[List[Dict]]:
        """
        Predict marginal probabilities for each label.
        
        Args:
            X: Feature sequences
            
        Returns:
            Marginal probabilities for each position
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict_marginals(X)
    
    def get_feature_weights(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top feature weights for each label.
        
        Args:
            top_n: Number of top features to return per label
            
        Returns:
            Dictionary mapping labels to top features
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        weights = {}
        
        for label in self.LABELS:
            # Get state features for this label
            state_features = []
            
            try:
                for feature, weight in self.model.state_features_.items():
                    if feature[0] == label:
                        state_features.append((feature[1], weight))
                
                # Sort by absolute weight
                state_features.sort(key=lambda x: abs(x[1]), reverse=True)
                weights[label] = state_features[:top_n]
            except:
                weights[label] = []
        
        return weights
    
    def print_feature_weights(self, top_n: int = 10):
        """Print top feature weights for each label."""
        weights = self.get_feature_weights(top_n)
        
        print("\n" + "="*60)
        print("TOP FEATURE WEIGHTS BY LABEL")
        print("="*60)
        
        for label in self.LABELS:
            print(f"\n{label}:")
            print("-" * 40)
            
            if label in weights and len(weights[label]) > 0:
                for feature, weight in weights[label]:
                    print(f"  {feature:30s} {weight:8.4f}")
            else:
                print("  (no features)")
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Output file path
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {save_path}")
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Model file path
        """
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from: {load_path}")
    
    def evaluate(self, X: List[List[Dict]], y_true: List[List[str]]) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y_true: True labels
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        y_pred = self.predict(X)
        
        # Overall metrics
        f1 = metrics.flat_f1_score(y_true, y_pred, 
                                   average='weighted', 
                                   labels=self.LABELS)
        
        precision = metrics.flat_precision_score(y_true, y_pred,
                                                 average='weighted',
                                                 labels=self.LABELS)
        
        recall = metrics.flat_recall_score(y_true, y_pred,
                                          average='weighted',
                                          labels=self.LABELS)
        
        accuracy = metrics.flat_accuracy_score(y_true, y_pred)
        
        results = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'samples': len(X)
        }
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Samples:   {results['samples']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        print("\nPer-Label Metrics:")
        print(metrics.flat_classification_report(
            y_true, y_pred, labels=self.LABELS, digits=3
        ))
        
        return results
