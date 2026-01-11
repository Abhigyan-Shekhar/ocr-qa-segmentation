"""
Postprocessing module for extracting question-answer pairs from CRF predictions.

Converts BIO tag sequences into structured QA pairs.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ocr_engine import OCRLine



@dataclass
class QuestionAnswerPair:
    """Represents a question-answer pair."""
    question_number: int
    question_text: str
    answer_text: str
    question_lines: List[int]  # Line indices
    answer_lines: List[int]    # Line indices
    confidence: float          # Average OCR confidence


class QAPairExtractor:
    """Extract question-answer pairs from tagged sequences."""
    
    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize extractor.
        
        Args:
            min_confidence: Minimum OCR confidence threshold
        """
        self.min_confidence = min_confidence
    
    def extract_pairs(self, lines: List[OCRLine], 
                     tags: List[str],
                     include_orphans: bool = False) -> List[QuestionAnswerPair]:
        """
        Extract question-answer pairs from tagged lines.
        
        Args:
            lines: OCR lines
            tags: Predicted tags (same length as lines)
            include_orphans: Whether to include questions without answers
            
        Returns:
            List of QuestionAnswerPair objects
        """
        if len(lines) != len(tags):
            raise ValueError("Lines and tags must have same length")
        
        # Group consecutive tags into segments
        segments = self._group_segments(lines, tags)
        
        # Pair questions with answers
        pairs = self._pair_questions_answers(segments, include_orphans)
        
        return pairs
    
    def _group_segments(self, lines: List[OCRLine], 
                       tags: List[str]) -> List[Dict]:
        """
        Group consecutive tags into segments.
        
        Args:
            lines: OCR lines
            tags: Tag sequence
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        current_segment = None
        
        for idx, (line, tag) in enumerate(zip(lines, tags)):
            if tag == 'O':
                # Skip 'Other' tags
                if current_segment is not None:
                    segments.append(current_segment)
                    current_segment = None
                continue
            
            segment_type = tag.split('-')[1]  # 'Q' or 'A'
            tag_prefix = tag.split('-')[0]    # 'B' or 'I'
            
            # Start new segment on B- tags or type change
            if tag_prefix == 'B' or current_segment is None or \
               current_segment['type'] != segment_type:
                if current_segment is not None:
                    segments.append(current_segment)
                
                current_segment = {
                    'type': segment_type,
                    'lines': [line],
                    'indices': [idx],
                    'tags': [tag]
                }
            else:
                # Continue current segment
                current_segment['lines'].append(line)
                current_segment['indices'].append(idx)
                current_segment['tags'].append(tag)
        
        # Add final segment
        if current_segment is not None:
            segments.append(current_segment)
        
        return segments
    
    def _merge_segment_text(self, segment: Dict) -> Tuple[str, float]:
        """
        Merge text from segment lines.
        
        Args:
            segment: Segment dictionary
            
        Returns:
            (merged_text, avg_confidence)
        """
        texts = [line.text.strip() for line in segment['lines']]
        merged = ' '.join(texts)
        
        confidences = [line.confidence for line in segment['lines']]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return merged, avg_conf
    
    def _pair_questions_answers(self, segments: List[Dict],
                                include_orphans: bool = False) -> List[QuestionAnswerPair]:
        """
        Pair question segments with answer segments.
        
        Args:
            segments: List of segments
            include_orphans: Include questions without answers
            
        Returns:
            List of QuestionAnswerPair objects
        """
        pairs = []
        question_number = 0
        
        i = 0
        while i < len(segments):
            segment = segments[i]
            
            if segment['type'] == 'Q':
                question_number += 1
                
                # Merge question text
                q_text, q_conf = self._merge_segment_text(segment)
                q_indices = segment['indices']
                
                # Look for following answer
                answer_text = ""
                a_indices = []
                a_conf = 0.0
                
                # Check next segment(s) for answers
                j = i + 1
                while j < len(segments) and segments[j]['type'] == 'A':
                    a_text, a_c = self._merge_segment_text(segments[j])
                    answer_text += (" " + a_text if answer_text else a_text)
                    a_indices.extend(segments[j]['indices'])
                    a_conf = (a_conf + a_c) / 2 if a_conf > 0 else a_c
                    j += 1
                
                # Create pair if answer exists or orphans allowed
                if answer_text or include_orphans:
                    avg_conf = (q_conf + a_conf) / 2 if a_conf > 0 else q_conf
                    
                    pair = QuestionAnswerPair(
                        question_number=question_number,
                        question_text=q_text.strip(),
                        answer_text=answer_text.strip(),
                        question_lines=q_indices,
                        answer_lines=a_indices,
                        confidence=avg_conf
                    )
                    
                    pairs.append(pair)
                
                # Skip the answer segments we processed
                i = j
            else:
                # Skip standalone answer segments
                i += 1
        
        return pairs
    
    def pairs_to_dict(self, pairs: List[QuestionAnswerPair]) -> List[Dict]:
        """
        Convert pairs to dictionary format.
        
        Args:
            pairs: List of QuestionAnswerPair objects
            
        Returns:
            List of dictionaries
        """
        return [
            {
                'question_number': pair.question_number,
                'question': pair.question_text,
                'answer': pair.answer_text,
                'question_lines': pair.question_lines,
                'answer_lines': pair.answer_lines,
                'confidence': round(pair.confidence, 3)
            }
            for pair in pairs
        ]
    
    def pairs_to_formatted_text(self, pairs: List[QuestionAnswerPair]) -> str:
        """
        Format pairs as human-readable text.
        
        Args:
            pairs: List of QuestionAnswerPair objects
            
        Returns:
            Formatted string
        """
        output = []
        output.append("=" * 70)
        output.append("EXTRACTED QUESTION-ANSWER PAIRS")
        output.append("=" * 70)
        
        for pair in pairs:
            output.append(f"\n{'─' * 70}")
            output.append(f"Question {pair.question_number}:")
            output.append(f"{'─' * 70}")
            output.append(f"{pair.question_text}")
            
            if pair.answer_text:
                output.append(f"\nAnswer:")
                output.append(f"{pair.answer_text}")
            else:
                output.append(f"\n[No answer provided]")
            
            output.append(f"\n(Confidence: {pair.confidence:.2%})")
        
        output.append(f"\n{'=' * 70}")
        output.append(f"Total: {len(pairs)} question-answer pairs")
        output.append("=" * 70)
        
        return '\n'.join(output)
    
    def filter_low_confidence(self, pairs: List[QuestionAnswerPair],
                             threshold: Optional[float] = None) -> List[QuestionAnswerPair]:
        """
        Filter out low-confidence pairs.
        
        Args:
            pairs: List of pairs
            threshold: Confidence threshold (uses self.min_confidence if None)
            
        Returns:
            Filtered list
        """
        threshold = threshold or self.min_confidence
        return [p for p in pairs if p.confidence >= threshold]
