#!/usr/bin/env python3
"""
TrOCR Fine-tuning Script for Kaggle (Tesla P100/T4)
Fine-tunes Microsoft TrOCR-base on CROHME + GNHK + FUNSD datasets
"""

import os
import subprocess
import glob
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# PHASE 1: ENVIRONMENT SETUP
# ============================================================================

def install_dependencies():
    """Install required packages"""
    subprocess.run([
        "pip", "install", "-q",
        "transformers>=4.30.0",
        "datasets>=2.14.0", 
        "jiwer>=3.0.0",
        "accelerate>=0.21.0",
        "torch>=2.0.0",
        "Pillow>=9.0.0",
        "pandas>=1.5.0",
        "tqdm>=4.65.0",
        "evaluate>=0.4.0"
    ], check=True)
    print("✓ Dependencies installed")

install_dependencies()

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from datasets import load_metric
import evaluate
from tqdm import tqdm

# ============================================================================
# PHASE 2: DATA PREPARATION - CROHME (Math Equations)
# ============================================================================

def prepare_crohme_data(output_dir: str = "./crohme_data") -> pd.DataFrame:
    """
    Clone CROHME extractor, run extraction, generate metadata.csv
    Returns DataFrame with (image_path, latex_label)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone the CROHME extractor repository
    repo_url = "https://github.com/ThomasLech/CROHME_extractor.git"
    repo_dir = os.path.join(output_dir, "CROHME_extractor")
    
    if not os.path.exists(repo_dir):
        print("Cloning CROHME extractor...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    # Navigate to repo and run extraction
    extract_script = os.path.join(repo_dir, "extract.py")
    inkml_dir = os.path.join(repo_dir, "data")
    png_output_dir = os.path.join(output_dir, "images")
    
    os.makedirs(png_output_dir, exist_ok=True)
    
    # Run extraction if images don't exist
    if len(glob.glob(os.path.join(png_output_dir, "*.png"))) < 100:
        print("Extracting CROHME InkML to PNG...")
        os.chdir(repo_dir)
        
        # The extractor expects specific structure - run it
        try:
            subprocess.run([
                "python", "extract.py",
                "--inkml_dir", "./data",
                "--output_dir", png_output_dir
            ], check=True, timeout=600)
        except subprocess.TimeoutExpired:
            print("Extraction timed out, using available data")
        except Exception as e:
            print(f"Extraction error: {e}, attempting manual extraction...")
            # Manual extraction fallback
            manual_extract_inkml(repo_dir, png_output_dir)
        
        os.chdir("/kaggle/working")
    
    # Build metadata from extracted images and labels
    metadata = []
    
    # Parse labels from InkML files
    inkml_files = glob.glob(os.path.join(repo_dir, "data", "**", "*.inkml"), recursive=True)
    
    for inkml_path in tqdm(inkml_files, desc="Parsing CROHME labels"):
        try:
            with open(inkml_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract LaTeX annotation
            import re
            match = re.search(r'<annotation type="truth">(.*?)</annotation>', content, re.DOTALL)
            if match:
                latex_label = match.group(1).strip()
                # Clean up the label
                latex_label = latex_label.replace('$', '').strip()
                
                # Find corresponding PNG
                base_name = Path(inkml_path).stem
                png_path = os.path.join(png_output_dir, f"{base_name}.png")
                
                if os.path.exists(png_path):
                    metadata.append({
                        "image_path": png_path,
                        "text": latex_label,
                        "source": "crohme"
                    })
        except Exception as e:
            continue
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "crohme_metadata.csv"), index=False)
    print(f"✓ CROHME: {len(df)} samples prepared")
    return df


def manual_extract_inkml(repo_dir: str, output_dir: str):
    """Fallback manual InkML to PNG extraction"""
    from PIL import Image, ImageDraw
    import xml.etree.ElementTree as ET
    
    inkml_files = glob.glob(os.path.join(repo_dir, "data", "**", "*.inkml"), recursive=True)
    
    for inkml_path in tqdm(inkml_files[:2000], desc="Manual extraction"):
        try:
            tree = ET.parse(inkml_path)
            root = tree.getroot()
            
            # Get all trace elements (strokes)
            ns = {'inkml': 'http://www.w3.org/2003/InkML'}
            traces = root.findall('.//inkml:trace', ns) or root.findall('.//trace')
            
            if not traces:
                continue
            
            all_points = []
            strokes = []
            
            for trace in traces:
                points = []
                coords = trace.text.strip().split(',')
                for coord in coords:
                    parts = coord.strip().split()
                    if len(parts) >= 2:
                        x, y = float(parts[0]), float(parts[1])
                        points.append((x, y))
                        all_points.append((x, y))
                if points:
                    strokes.append(points)
            
            if not all_points:
                continue
            
            # Normalize and create image
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            width = max(int(max_x - min_x) + 40, 100)
            height = max(int(max_y - min_y) + 40, 50)
            
            img = Image.new('RGB', (width, height), 'white')
            draw = ImageDraw.Draw(img)
            
            for stroke in strokes:
                normalized = [(int(x - min_x + 20), int(y - min_y + 20)) for x, y in stroke]
                if len(normalized) > 1:
                    draw.line(normalized, fill='black', width=2)
            
            base_name = Path(inkml_path).stem
            img.save(os.path.join(output_dir, f"{base_name}.png"))
            
        except Exception as e:
            continue


# ============================================================================
# PHASE 3: DATA PREPARATION - GNHK (Handwriting)
# ============================================================================

def prepare_gnhk_data(output_dir: str = "./gnhk_data") -> pd.DataFrame:
    """
    Download and prepare GNHK dataset
    Returns DataFrame with (image_path, text_label)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    repo_url = "https://github.com/GoodNotes/GNHK-dataset.git"
    repo_dir = os.path.join(output_dir, "GNHK-dataset")
    
    if not os.path.exists(repo_dir):
        print("Cloning GNHK dataset...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
    
    metadata = []
    
    # Parse JSON annotations
    json_files = glob.glob(os.path.join(repo_dir, "**", "*.json"), recursive=True)
    
    for json_path in tqdm(json_files, desc="Parsing GNHK annotations"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get the base image path
            img_dir = os.path.dirname(json_path)
            
            # Extract word-level annotations
            if isinstance(data, dict):
                for item in data.get('words', []):
                    if 'text' in item and 'image' in item:
                        img_path = os.path.join(img_dir, item['image'])
                        if os.path.exists(img_path):
                            metadata.append({
                                "image_path": img_path,
                                "text": item['text'],
                                "source": "gnhk"
                            })
                
                # Also check for lines
                for item in data.get('lines', []):
                    if 'text' in item and 'image' in item:
                        img_path = os.path.join(img_dir, item['image'])
                        if os.path.exists(img_path):
                            metadata.append({
                                "image_path": img_path,
                                "text": item['text'],
                                "source": "gnhk"
                            })
        except Exception as e:
            continue
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "gnhk_metadata.csv"), index=False)
    print(f"✓ GNHK: {len(df)} samples prepared")
    return df


# ============================================================================
# PHASE 4: DATA PREPARATION - FUNSD (Layout/Validation)
# ============================================================================

def prepare_funsd_data(output_dir: str = "./funsd_data") -> pd.DataFrame:
    """
    Download and prepare FUNSD dataset for validation
    Returns DataFrame with (image_path, text_label)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    repo_url = "https://github.com/GuillaumeJaume/FUNSD.git"
    repo_dir = os.path.join(output_dir, "FUNSD")
    
    if not os.path.exists(repo_dir):
        print("Cloning FUNSD dataset...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, repo_dir], check=True)
    
    metadata = []
    
    # Parse training data annotations
    annotation_dir = os.path.join(repo_dir, "dataset", "training_data", "annotations")
    image_dir = os.path.join(repo_dir, "dataset", "training_data", "images")
    
    json_files = glob.glob(os.path.join(annotation_dir, "*.json"))
    
    for json_path in tqdm(json_files, desc="Parsing FUNSD annotations"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for form in data.get('form', []):
                text = form.get('text', '').strip()
                if text and len(text) > 2:
                    # For FUNSD, we use the full document image
                    base_name = Path(json_path).stem
                    img_path = os.path.join(image_dir, f"{base_name}.png")
                    
                    if os.path.exists(img_path):
                        # Create word-level crops for training
                        box = form.get('box', [])
                        if len(box) == 4:
                            metadata.append({
                                "image_path": img_path,
                                "text": text,
                                "box": box,
                                "source": "funsd"
                            })
        except Exception as e:
            continue
    
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(output_dir, "funsd_metadata.csv"), index=False)
    print(f"✓ FUNSD: {len(df)} samples prepared")
    return df


# ============================================================================
# PHASE 5: DATASET CLASS
# ============================================================================

class TrOCRDataset(Dataset):
    """
    Custom Dataset for TrOCR fine-tuning
    Handles image loading and label tokenization with proper padding masking
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
        image_size: Tuple[int, int] = (384, 384)
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_target_length = max_target_length
        self.image_size = image_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(row['image_path']).convert('RGB')
            
            # Handle FUNSD crops if box is available
            if 'box' in row and pd.notna(row.get('box')):
                box = row['box']
                if isinstance(box, str):
                    import ast
                    box = ast.literal_eval(box)
                if len(box) == 4:
                    image = image.crop((box[0], box[1], box[2], box[3]))
            
            # Resize to standard size
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            
        except Exception as e:
            # Return a blank image on error
            image = Image.new('RGB', self.image_size, 'white')
        
        # Process image
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Tokenize labels
        text = str(row['text'])
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Mask padding tokens with -100 for loss calculation
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


# ============================================================================
# PHASE 6: TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    model_name: str = "microsoft/trocr-base-handwritten"
    output_dir: str = "./tr_ocr_finetuned"
    
    # Training params
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 4e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # Mixed precision
    fp16: bool = True
    
    # Logging
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Generation
    max_target_length: int = 128
    num_beams: int = 4


def compute_metrics(pred):
    """Compute CER and WER metrics"""
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 with pad token id
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    # Compute metrics
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {
        "cer": cer,
        "wer": wer
    }


# ============================================================================
# PHASE 7: MAIN TRAINING LOOP
# ============================================================================

def main():
    print("=" * 60)
    print("TrOCR Fine-tuning Pipeline")
    print("=" * 60)
    
    config = TrainingConfig()
    
    # Step 1: Prepare all datasets
    print("\n[1/6] Preparing datasets...")
    
    crohme_df = prepare_crohme_data()
    gnhk_df = prepare_gnhk_data()
    funsd_df = prepare_funsd_data()
    
    # Combine datasets
    all_data = pd.concat([crohme_df, gnhk_df], ignore_index=True)
    
    # Filter out samples with very short or very long labels
    all_data = all_data[all_data['text'].str.len() > 1]
    all_data = all_data[all_data['text'].str.len() < 200]
    
    # Shuffle and split
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    split_idx = int(len(all_data) * 0.9)
    train_df = all_data[:split_idx]
    val_df = pd.concat([all_data[split_idx:], funsd_df], ignore_index=True)
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    
    # Step 2: Load processor and model
    print("\n[2/6] Loading TrOCR model and processor...")
    
    global processor
    processor = TrOCRProcessor.from_pretrained(config.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
    
    # Configure model for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = config.max_target_length
    model.config.num_beams = config.num_beams
    
    print(f"   Model loaded: {config.model_name}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Create datasets
    print("\n[3/6] Creating datasets...")
    
    train_dataset = TrOCRDataset(
        df=train_df,
        processor=processor,
        max_target_length=config.max_target_length
    )
    
    val_dataset = TrOCRDataset(
        df=val_df,
        processor=processor,
        max_target_length=config.max_target_length
    )
    
    print(f"   Train dataset: {len(train_dataset)} samples")
    print(f"   Val dataset: {len(val_dataset)} samples")
    
    # Step 4: Configure training
    print("\n[4/6] Configuring training...")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        
        # Training
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        
        # Mixed precision
        fp16=config.fp16,
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        
        # Logging
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        
        # Generation during eval
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.num_beams,
        
        # Misc
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Step 5: Create trainer
    print("\n[5/6] Initializing trainer...")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Step 6: Train!
    print("\n[6/6] Starting training...")
    print("-" * 40)
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    
    print("=" * 60)
    print(f"✓ Training complete!")
    print(f"✓ Model saved to: {config.output_dir}")
    print("=" * 60)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    metrics = trainer.evaluate()
    print(f"   CER: {metrics.get('eval_cer', 'N/A'):.4f}")
    print(f"   WER: {metrics.get('eval_wer', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
