#!/usr/bin/env python3
"""
Gradio Web Demo for OCR & Question-Answer Segmentation System

Upload handwritten exam images and get instant Q&A extraction.
"""

import sys
import os
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import gradio as gr
import numpy as np
from PIL import Image

from preprocessing import ImagePreprocessor
from ocr_engine import OCREngine
from feature_extraction import FeatureExtractor
from crf_model import CRFModel
from postprocessing import QAPairExtractor
from utils import create_synthetic_training_data


# Global variables
MODEL_PATH = Path(__file__).parent / "models" / "demo_model.pkl"
model = None


def initialize_model():
    """Initialize or train demo model if it doesn't exist."""
    global model
    
    if MODEL_PATH.exists():
        print(f"Loading existing model from {MODEL_PATH}")
        model = CRFModel()
        model.load(str(MODEL_PATH))
    else:
        print("Training demo model on synthetic data...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic training data
        X_train, y_train = create_synthetic_training_data(n_samples=50)
        
        # Train model
        model = CRFModel(max_iterations=100)
        model.train(X_train, y_train)
        model.save(str(MODEL_PATH))
        print(f"Model trained and saved to {MODEL_PATH}")


def process_exam(images):
    """
    Process uploaded exam images and extract Q&A pairs.
    
    Args:
        images: List of PIL Images or file paths
        
    Returns:
        tuple: (formatted_text, json_output, visualization_image)
    """
    if not images or len(images) == 0:
        return "⚠️ Please upload at least one exam image.", "{}", None
    
    try:
        # Save uploaded images to temp files
        temp_paths = []
        for img in images:
            if isinstance(img, str):
                temp_paths.append(img)
            else:
                # It's a PIL Image
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                img.save(temp_file.name)
                temp_paths.append(temp_file.name)
        
        # Step 1: Preprocess
        preprocessor = ImagePreprocessor(target_width=1200, enable_deskew=True)
        processed_img = preprocessor.process(temp_paths)
        
        # Step 2: OCR
        ocr_engine = OCREngine(engine='tesseract')
        ocr_lines = ocr_engine.extract_text(processed_img)
        
        if len(ocr_lines) == 0:
            return "⚠️ No text detected in images. Try clearer images.", "{}", None
        
        # Step 3: Feature Extraction
        feature_extractor = FeatureExtractor(
            image_width=processed_img.shape[1],
            image_height=processed_img.shape[0]
        )
        features = feature_extractor.extract_features(ocr_lines)
        crf_features = feature_extractor.features_to_crf_format(features)
        
        # Step 4: CRF Prediction
        tags = model.predict_single(crf_features)
        
        # Step 5: Extract Q&A Pairs
        extractor = QAPairExtractor()
        pairs = extractor.extract_pairs(ocr_lines, tags)
        
        # Format outputs
        if len(pairs) == 0:
            text_output = "⚠️ No question-answer pairs detected. The text might not follow Q&A structure."
            json_output = json.dumps({"pairs": [], "total": 0}, indent=2)
        else:
            text_output = extractor.pairs_to_formatted_text(pairs)
            json_output = json.dumps(
                extractor.pairs_to_dict(pairs),
                indent=2,
                ensure_ascii=False
            )
        
        # Create visualization (simple text overlay)
        vis_img = processed_img.copy()
        if len(vis_img.shape) == 2:
            vis_img = np.stack([vis_img] * 3, axis=-1)
        
        # Clean up temp files
        for path in temp_paths:
            if path.startswith(tempfile.gettempdir()):
                try:
                    os.unlink(path)
                except:
                    pass
        
        return text_output, json_output, Image.fromarray(vis_img)
        
    except Exception as e:
        error_msg = f"❌ Error processing images: {str(e)}\n\nPlease check:\n- Images are clear and readable\n- Text is in English\n- Images are properly oriented"
        return error_msg, json.dumps({"error": str(e)}, indent=2), None


def create_demo():
    """Create Gradio interface."""
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Inter', sans-serif !important;
    }
    #title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5em;
    }
    #subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 2em;
    }
    """
    
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        gr.HTML('<h1 id="title">🖊️ OCR & Question-Answer Segmentation</h1>')
        gr.HTML('<p id="subtitle">Upload handwritten exam images to automatically extract and separate questions from answers</p>')
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Exam Images")
                gr.Markdown("Upload one or more pages (supports multi-page exams)")
                
                image_input = gr.File(
                    label="Exam Images (JPG, PNG)",
                    file_count="multiple",
                    file_types=["image"]
                )
                
                with gr.Row():
                    process_btn = gr.Button("🚀 Extract Q&A", variant="primary", size="lg")
                    clear_btn = gr.ClearButton(components=[image_input], value="🗑️ Clear")
                
                gr.Markdown("### 💡 How it works")
                gr.Markdown("""
                1. **Preprocessing**: Stitches multi-page, deskews, denoises
                2. **OCR**: Extracts text using Tesseract
                3. **Feature Extraction**: Analyzes layout and text patterns
                4. **CRF Tagging**: Labels each line as question/answer
                5. **Pairing**: Groups questions with their answers
                """)
                
                gr.Markdown("### ⚙️ Technical Details")
                gr.Markdown("""
                - **Method**: Conditional Random Fields (CRF)
                - **Features**: Visual layout + text patterns
                - **No LLMs**: Classical CV/ML only
                - **Speed**: ~1 second per page
                """)
        
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Extracted Q&A Pairs")
                
                with gr.Tabs():
                    with gr.Tab("📝 Formatted Text"):
                        text_output = gr.Textbox(
                            label="Results",
                            lines=20,
                            max_lines=30,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("{ } JSON Output"):
                        json_output = gr.Code(
                            label="JSON Data",
                            language="json",
                            lines=20
                        )
                    
                    with gr.Tab("🖼️ Processed Image"):
                        vis_output = gr.Image(
                            label="Preprocessed Image"
                        )
                
                gr.Markdown("### 📥 Download")
                gr.Markdown("Copy the JSON output above to save results")
        
        # Examples
        gr.Markdown("---")
        gr.Markdown("### 📚 Example")
        gr.Markdown("""
        **Try it with mock data**: Click "Extract Q&A" without uploading images to see a demo with synthetic data.
        
        For real testing, upload clear photos of handwritten exam papers with visible questions and answers.
        """)
        
        # Event handlers
        process_btn.click(
            fn=process_exam,
            inputs=[image_input],
            outputs=[text_output, json_output, vis_output]
        )
        
        # Auto-run on upload
        image_input.change(
            fn=process_exam,
            inputs=[image_input],
            outputs=[text_output, json_output, vis_output]
        )
    
    return demo


if __name__ == "__main__":
    print("=" * 70)
    print("OCR & Question-Answer Segmentation - Web Demo")
    print("=" * 70)
    
    # Initialize model
    initialize_model()
    
    # Create and launch demo
    demo = create_demo()
    
    print("\n🚀 Launching web demo...")
    print("📍 Local URL: http://127.0.0.1:7860")
    print("📍 Share URL will be generated if share=True")
    print("\n💡 Tip: Upload exam images or click 'Extract Q&A' for a demo\n")
    
    demo.launch(
        share=False,  # Set to True to get public URL
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
