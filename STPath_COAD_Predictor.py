#!/usr/bin/env python3
"""
STPath-COAD: Colorectal Cancer Cell Proportion Predictor
=======================================================

A comprehensive tool for predicting cell type proportions in colorectal cancer H&E images
using multiple foundation models and XGBoost regression. This script provides an easy-to-use 
interface for any users.

Author: Saishi Cui
Date: Sept 2025

Purpose: Main prediction script for STPath-COAD framework that combines multiple 
foundation models (Conch, UNI2h, ProvGigapath, Virchow, Virchow2) to predict 
cell type proportions in colorectal cancer H&E whole slide images.
"""

import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import timm
from conch.open_clip_custom import create_model_from_pretrained
from huggingface_hub import login
from datetime import datetime
from timm.layers import SwiGLUPacked
import xgboost as xgb
from torchvision import transforms
import pickle
import os
import argparse
import json
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class STPath_COAD_Predictor:
    """
    Main class for predicting cell type proportions in colorectal cancer H&E images
    """
    
    def __init__(self, hf_token, patch_size=240, device=None, white_threshold=220, white_ratio_cutoff=0.4):
        """
        Initialize the predictor with foundation models and parameters
        
        Args:
            hf_token (str): HuggingFace token for accessing foundation models (get from https://huggingface.co/settings/tokens)
            patch_size (int): Size of patches to extract from WSI (default: 240)
            device (str): Device to use ('auto', 'cpu', 'cuda', 'mps')
            white_threshold (int): Threshold for white pixel detection (default: 220)
            white_ratio_cutoff (float): Ratio cutoff for white patch detection (default: 0.4)
        """
        self.hf_token = hf_token
        self.patch_size = patch_size
        self.device = self._setup_device(device)
        
        # Initialize models and transforms
        self.models = {}
        self.transforms = {}
        self.important_features = {}
        self.xgb_models = {}
        
        # Cell types
        self.cell_types = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T cells", "Other Immune Cells"]
        
        # White patch detection parameters
        self.white_threshold = white_threshold
        self.white_ratio_cutoff = white_ratio_cutoff
        
        print(f"Initializing STPath-COAD Predictor...")
        print(f"Patch size: {patch_size}x{patch_size} pixels")
        print(f"Device: {self.device}")
        print(f"White threshold: {white_threshold}")
        print(f"White ratio cutoff: {white_ratio_cutoff}")
        
    def _setup_device(self, device):
        """Setup the appropriate device for model inference"""
        if device == 'auto' or device is None:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        elif device == 'cuda' and torch.cuda.is_available():
            return torch.device("cuda")
        elif device == 'mps' and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _log(self, message):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def load_foundation_models(self):
        """Load all foundation models"""
        self._log("Loading foundation models...")
        
        # Login to HuggingFace
        login(token=self.hf_token)
        
        # Load Conch model
        self._log("Loading Conch model...")
        model_conch, transform_conch = create_model_from_pretrained(
            'conch_ViT-B-16', 
            "hf_hub:MahmoodLab/conch", 
            hf_auth_token=self.hf_token
        )
        self.models['conch'] = model_conch.eval().to(self.device)
        self.transforms['conch'] = transform_conch
        
        # Load UNI2h model
        self._log("Loading UNI2h model...")
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model_uni2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        self.models['uni2h'] = model_uni2h.eval().to(self.device)
        
        # Load ProvGigapath model
        self._log("Loading ProvGigapath model...")
        model_provgigapath = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.models['provgigapath'] = model_provgigapath.eval().to(self.device)
        
        # Load Virchow model
        self._log("Loading Virchow model...")
        model_virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, 
                                        mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        self.models['virchow'] = model_virchow.eval().to(self.device)
        
        # Load Virchow2 model
        self._log("Loading Virchow2 model...")
        model_virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, 
                                         mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
        self.models['virchow2'] = model_virchow2.eval().to(self.device)
        
        # Define standard transforms
        self.transforms['standard'] = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self.transforms['provgigapath'] = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        self._log("All foundation models loaded successfully!")
    
    def load_xgboost_models(self, model_paths, important_features_paths):
        """
        Load XGBoost models and important features
        
        Args:
            model_paths (dict): Dictionary mapping cell types to XGBoost model paths
            important_features_paths (dict): Dictionary mapping cell types to important features paths
        """
        self._log("Loading XGBoost models and important features...")
        
        for cell_type in self.cell_types:
            # Load important features
            if cell_type in important_features_paths:
                features_path = important_features_paths[cell_type]
                self._log(f"Loading important features for {cell_type} from {features_path}")
                self.important_features[cell_type] = pickle.load(open(features_path, "rb"))
            
            # Load XGBoost model
            if cell_type in model_paths:
                model_path = model_paths[cell_type]
                self._log(f"Loading XGBoost model for {cell_type} from {model_path}")
                xgb_model = xgb.Booster()
                xgb_model.load_model(model_path)
                self.xgb_models[cell_type] = xgb_model
        
        self._log("All XGBoost models and features loaded successfully!")
    
    def create_patches(self, image_path):
        """
        Create patches from WSI image
        
        Args:
            image_path (str): Path to the WSI image
            
        Returns:
            list: List of patch information dictionaries
        """
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(image_path)
        image_np = np.array(image)
        
        height, width = image_np.shape[:2]
        self._log(f"Original image size: {width} x {height}")
        
        # Calculate how many patches we can fit
        num_patches_width = width // self.patch_size
        num_patches_height = height // self.patch_size
        
        self._log(f"Creating {num_patches_height} x {num_patches_width} = {num_patches_height * num_patches_width} patches")
        self._log(f"Each patch size: {self.patch_size} x {self.patch_size} pixels")
        
        # Calculate discarded border sizes
        discarded_width = width % self.patch_size
        discarded_height = height % self.patch_size
        self._log(f"Discarded border: {discarded_width} pixels (width) x {discarded_height} pixels (height)")
        
        patch_list = []
        
        for row in range(num_patches_height):
            for col in range(num_patches_width):
                y_start = row * self.patch_size
                y_end = y_start + self.patch_size
                x_start = col * self.patch_size
                x_end = x_start + self.patch_size
                
                patch_np = image_np[y_start:y_end, x_start:x_end]
                patch_image = Image.fromarray(patch_np)
                patch_id = f"row{row:03d}_col{col:03d}"
                
                patch_info = {
                    'patch_id': patch_id,
                    'row': row,
                    'col': col,
                    'x_start': x_start,
                    'y_start': y_start,
                    'x_end': x_end,
                    'y_end': y_end,
                    'patch_width': self.patch_size,
                    'patch_height': self.patch_size,
                    'patch_image': patch_image,
                    'patch_array': patch_np
                }
                
                patch_list.append(patch_info)
        
        self._log(f"Successfully created {len(patch_list)} patches")
        return patch_list
    
    def is_white_patch(self, patch_array):
        """Check if a patch is too white (likely background)"""
        # Convert to grayscale if RGB
        if len(patch_array.shape) == 3:
            gray = np.mean(patch_array, axis=2)
        else:
            gray = patch_array
        white_pixels = np.sum(gray > self.white_threshold)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels
        return white_ratio > self.white_ratio_cutoff
    
    def extract_features(self, patch_image):
        """Extract features from all foundation models for a single patch"""
        with torch.no_grad():
            # UNI2h features  
            img_uni2h = self.transforms['standard'](patch_image).unsqueeze(0).to(self.device)
            features_uni2h_full = self.models['uni2h'](img_uni2h)
            features_uni2h_full = features_uni2h_full.cpu().numpy().flatten()
            
            # Virchow features
            img_virchow = self.transforms['standard'](patch_image).unsqueeze(0).to(self.device)
            embeddings_virchow = self.models['virchow'](img_virchow)
            features_virchow_full = torch.cat([embeddings_virchow[:,0], embeddings_virchow[:,5:].mean(1)], dim=-1)
            features_virchow_full = features_virchow_full.cpu().numpy().flatten()
            
            # Virchow2 features
            img_virchow2 = self.transforms['standard'](patch_image).unsqueeze(0).to(self.device)
            embeddings_virchow2 = self.models['virchow2'](img_virchow2)
            features_virchow2_full = torch.cat([embeddings_virchow2[:,0], embeddings_virchow2[:,5:].mean(1)], dim=-1)
            features_virchow2_full = features_virchow2_full.cpu().numpy().flatten()
            
            # ProvGigapath features
            img_provgigapath = self.transforms['provgigapath'](patch_image).unsqueeze(0).to(self.device)
            features_provgigapath_full = self.models['provgigapath'](img_provgigapath)
            features_provgigapath_full = features_provgigapath_full.cpu().numpy().flatten()
            
            # Conch features
            img_conch = self.transforms['conch'](patch_image).unsqueeze(0).to(self.device)
            features_conch_full = self.models['conch'].encode_image(img_conch, proj_contrast=False, normalize=False)
            features_conch_full = features_conch_full.cpu().numpy().flatten()
        
        return {
            'uni2h': features_uni2h_full,
            'virchow': features_virchow_full,
            'virchow2': features_virchow2_full,
            'provgigapath': features_provgigapath_full,
            'conch': features_conch_full
        }
    
    def predict_cell_proportions(self, patch_list):
        """
        Predict cell type proportions for all patches
        
        Args:
            patch_list (list): List of patch information dictionaries
            
        Returns:
            list: List of dictionaries with patch info and cell type proportions
        """
        self._log(f"Predicting cell type proportions for {len(patch_list)} patches...")
        
        results = []
        
        for i, patch_info in enumerate(tqdm(patch_list, desc="Processing patches")):
            
            # Check if patch is too white
            if self.is_white_patch(patch_info['patch_array']):
                # For white patches, assign None values
                results.append({
                    'patch_id': patch_info['patch_id'],
                    'row': patch_info['row'],
                    'col': patch_info['col'],
                    'x_start': patch_info['x_start'],
                    'y_start': patch_info['y_start'],
                    'x_end': patch_info['x_end'],
                    'y_end': patch_info['y_end'],
                    'cancer_proportion': None,
                    'stromal_proportion': None,
                    'normal_epithelia_proportion': None,
                    'tcells_proportion': None,
                    'other_immune_proportion': None,
                    'normalized_proportions': None,
                    'is_white_patch': True
                })
                continue
            
            # Extract features
            features = self.extract_features(patch_info['patch_image'])
            
            # Predict each cell type
            predictions = {}
            
            for cell_type in self.cell_types:
                # Get important feature indices for this cell type
                UNI2h_important_indices = [int(f.replace('f', '')) for f in self.important_features[cell_type]["UNI2h"]]
                Virchow_important_indices = [int(f.replace('f', '')) for f in self.important_features[cell_type]["Virchow"]]
                Virchow2_important_indices = [int(f.replace('f', '')) for f in self.important_features[cell_type]["Virchow2"]]
                ProvGigapath_important_indices = [int(f.replace('f', '')) for f in self.important_features[cell_type]["ProvGigapath"]]
                Conch_important_indices = [int(f.replace('f', '')) for f in self.important_features[cell_type]["Conch"]]
                
                # Select important features
                UNI2h_selected = features['uni2h'][UNI2h_important_indices]
                Virchow_selected = features['virchow'][Virchow_important_indices]
                Virchow2_selected = features['virchow2'][Virchow2_important_indices]
                ProvGigapath_selected = features['provgigapath'][ProvGigapath_important_indices]
                Conch_selected = features['conch'][Conch_important_indices]
                
                # Stack features in the specified order: UNI2h, Virchow, Virchow2, ProvGigapath, Conch
                combined_features = np.concatenate([
                    UNI2h_selected,
                    Virchow_selected,
                    Virchow2_selected,
                    ProvGigapath_selected,
                    Conch_selected
                ])
                
                # Prepare data for XGBoost prediction
                dtest = xgb.DMatrix(combined_features.reshape(1, -1))
                
                # Make prediction
                prediction = self.xgb_models[cell_type].predict(dtest)[0]
                predictions[cell_type] = float(prediction)
            
            # Normalize predictions to sum to 1
            total_prediction = sum(predictions.values())
            if total_prediction > 0:
                normalized_predictions = {cell_type: pred / total_prediction for cell_type, pred in predictions.items()}
            else:
                # If all predictions are 0, assign equal proportions
                normalized_predictions = {cell_type: 0.2 for cell_type in self.cell_types}
            
            # Store results
            results.append({
                'patch_id': patch_info['patch_id'],
                'row': patch_info['row'],
                'col': patch_info['col'],
                'x_start': patch_info['x_start'],
                'y_start': patch_info['y_start'],
                'x_end': patch_info['x_end'],
                'y_end': patch_info['y_end'],
                'cancer_proportion': normalized_predictions["Cancer Cells"],
                'stromal_proportion': normalized_predictions["Stromal Cells"],
                'normal_epithelia_proportion': normalized_predictions["Normal Epithelial Cells"],
                'tcells_proportion': normalized_predictions["T cells"],
                'other_immune_proportion': normalized_predictions["Other Immune Cells"],
                'normalized_proportions': [
                    normalized_predictions["Cancer Cells"],
                    normalized_predictions["Stromal Cells"],
                    normalized_predictions["Normal Epithelial Cells"],
                    normalized_predictions["T cells"],
                    normalized_predictions["Other Immune Cells"]
                ],
                'is_white_patch': False
            })
        
        self._log(f"Completed prediction for {len(results)} patches")
        return results
    
    def get_overall_proportions(self, results):
        """
        Calculate overall cell type proportions from prediction results
        
        Args:
            results (dict): Results dictionary from predict_image function
            
        Returns:
            dict: Overall proportions for each cell type
        """
        if 'predictions' not in results:
            raise ValueError("Results dictionary must contain 'predictions' key")
        
        # Filter out white patches
        valid_predictions = [r for r in results['predictions'] if not r['is_white_patch']]
        
        if not valid_predictions:
            return {
                'Cancer Cells': 0.0,
                'Stromal Cells': 0.0,
                'Normal Epithelial Cells': 0.0,
                'T Cells': 0.0,
                'Other Immune Cells': 0.0,
                'total_valid_patches': 0
            }
        
        # Calculate mean proportions
        cancer_props = [r['cancer_proportion'] for r in valid_predictions]
        stromal_props = [r['stromal_proportion'] for r in valid_predictions]
        normal_props = [r['normal_epithelia_proportion'] for r in valid_predictions]
        tcell_props = [r['tcells_proportion'] for r in valid_predictions]
        immune_props = [r['other_immune_proportion'] for r in valid_predictions]
        
        overall_proportions = {
            'Cancer Cells': np.mean(cancer_props),
            'Stromal Cells': np.mean(stromal_props),
            'Normal Epithelial Cells': np.mean(normal_props),
            'T Cells': np.mean(tcell_props),
            'Other Immune Cells': np.mean(immune_props),
            'total_valid_patches': len(valid_predictions)
        }
        
        # Ensure proportions sum to 1 (normalize)
        total = sum([overall_proportions[key] for key in overall_proportions.keys() if key != 'total_valid_patches'])
        if total > 0:
            for key in overall_proportions.keys():
                if key != 'total_valid_patches':
                    overall_proportions[key] = overall_proportions[key] / total
        
        return overall_proportions
    
    def predict_image(self, image_path, output_path=None):
        """
        Main function to predict cell proportions for a WSI image
        
        Args:
            image_path (str): Path to the WSI image
            output_path (str): Path to save results (optional)
            
        Returns:
            dict: Results dictionary with predictions and metadata
        """
        self._log(f"Starting prediction for image: {image_path}")
        
        # Create patches
        patch_list = self.create_patches(image_path)
        
        # Predict cell proportions
        results = self.predict_cell_proportions(patch_list)
        
        # Prepare output
        output_data = {
            'image_path': image_path,
            'patch_size': self.patch_size,
            'device': str(self.device),
            'total_patches': len(results),
            'valid_patches': len([r for r in results if not r['is_white_patch']]),
            'white_patches': len([r for r in results if r['is_white_patch']]),
            'cell_types': self.cell_types,
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results if output path is provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            # Save as CSV for easy analysis
            csv_path = output_path.with_suffix('.csv')
            df_data = []
            for result in results:
                if not result['is_white_patch']:
                    df_data.append({
                        'patch_id': result['patch_id'],
                        'row': result['row'],
                        'col': result['col'],
                        'x_start': result['x_start'],
                        'y_start': result['y_start'],
                        'x_end': result['x_end'],
                        'y_end': result['y_end'],
                        'cancer_proportion': result['cancer_proportion'],
                        'stromal_proportion': result['stromal_proportion'],
                        'normal_epithelia_proportion': result['normal_epithelia_proportion'],
                        'tcells_proportion': result['tcells_proportion'],
                        'other_immune_proportion': result['other_immune_proportion']
                    })
            
            if df_data:
                import pandas as pd
                df = pd.DataFrame(df_data)
                df.to_csv(csv_path, index=False)
            
            self._log(f"Results saved to {json_path} and {csv_path}")
        
        return output_data

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='CRC Cell Proportion Predictor')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--image_path', type=str, help='Path to H&E image')
    parser.add_argument('--hf_token', type=str, help='HuggingFace token')
    parser.add_argument('--model_dir', type=str, help='Directory containing XGBoost models')
    parser.add_argument('--features_dir', type=str, help='Directory containing important features files')
    parser.add_argument('--output_path', type=str, help='Output path for results')
    parser.add_argument('--patch_size', type=int, help='Patch size in pixels')
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], 
                       help='Device to use')
    parser.add_argument('--white_threshold', type=int, help='White pixel threshold')
    parser.add_argument('--white_ratio_cutoff', type=float, help='White ratio cutoff')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            return
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.image_path:
        config['image_path'] = args.image_path
    if args.hf_token:
        config['huggingface'] = {'token': args.hf_token}
    if args.model_dir:
        config['models'] = {'xgboost_dir': args.model_dir}
    if args.features_dir:
        config['models'] = config.get('models', {})
        config['models']['features_dir'] = args.features_dir
    if args.output_path:
        config['paths'] = config.get('paths', {})
        config['paths']['output_dir'] = args.output_path
    if args.patch_size:
        config['processing'] = config.get('processing', {})
        config['processing']['patch_size'] = args.patch_size
    if args.device:
        config['processing'] = config.get('processing', {})
        config['processing']['device'] = args.device
    if args.white_threshold:
        config['processing'] = config.get('processing', {})
        config['processing']['white_threshold'] = args.white_threshold
    if args.white_ratio_cutoff:
        config['processing'] = config.get('processing', {})
        config['processing']['white_ratio_cutoff'] = args.white_ratio_cutoff
    
    # Validate required configuration
    if not config.get('image_path'):
        print("Error: Image path is required. Use --image_path or specify in config file.")
        return
    
    if not config.get('huggingface', {}).get('token'):
        print("Error: HuggingFace token is required. Use --hf_token or specify in config file.")
        return
    
    if not config.get('models', {}).get('xgboost_dir'):
        print("Error: XGBoost model directory is required. Use --model_dir or specify in config file.")
        return
    
    if not config.get('models', {}).get('features_dir'):
        print("Error: Features directory is required. Use --features_dir or specify in config file.")
        return
    
    # Get configuration values with defaults
    image_path = config['image_path']
    hf_token = config['huggingface']['token']
    model_dir = config['models']['xgboost_dir']
    features_dir = config['models']['features_dir']
    output_path = config.get('paths', {}).get('output_dir')
    
    processing = config.get('processing', {})
    patch_size = processing.get('patch_size', 240)
    device = processing.get('device', 'auto')
    white_threshold = processing.get('white_threshold', 220)
    white_ratio_cutoff = processing.get('white_ratio_cutoff', 0.4)
    
    # Setup model paths
    cell_types = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "T cells", "Other Immune Cells"]
    model_paths = {}
    important_features_paths = {}
    
    for cell_type in cell_types:
        # XGBoost model path
        model_filename = f"xgboost_model_{cell_type.replace(' ', '_')}.json"
        model_path = os.path.join(model_dir, model_filename)
        if os.path.exists(model_path):
            model_paths[cell_type] = model_path
        else:
            print(f"Warning: Model file not found: {model_path}")
        
        # Important features path
        features_filename = f"important_features_{cell_type.replace(' ', '_')}.pkl"
        features_path = os.path.join(features_dir, features_filename)
        if os.path.exists(features_path):
            important_features_paths[cell_type] = features_path
        else:
            print(f"Warning: Features file not found: {features_path}")
    
    # Check if all required files exist
    if len(model_paths) != len(cell_types) or len(important_features_paths) != len(cell_types):
        print("Error: Not all required model and feature files found!")
        return
    
    # Initialize predictor
    predictor = STPath_COAD_Predictor(
        hf_token=hf_token,
        patch_size=patch_size,
        device=device,
        white_threshold=white_threshold,
        white_ratio_cutoff=white_ratio_cutoff
    )
    
    # Load models
    predictor.load_foundation_models()
    predictor.load_xgboost_models(model_paths, important_features_paths)
    
    # Run prediction
    results = predictor.predict_image(image_path, output_path)
    
    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"Image: {results['image_path']}")
    print(f"Total patches: {results['total_patches']}")
    print(f"Valid patches: {results['valid_patches']}")
    print(f"White patches (excluded): {results['white_patches']}")
    print(f"Patch size: {results['patch_size']}x{results['patch_size']} pixels")
    print(f"Device used: {results['device']}")
    
    if results['valid_patches'] > 0:
        # Calculate average proportions
        valid_results = [r for r in results['predictions'] if not r['is_white_patch']]
        avg_proportions = {
            'Cancer Cells': np.mean([r['cancer_proportion'] for r in valid_results]),
            'Stromal Cells': np.mean([r['stromal_proportion'] for r in valid_results]),
            'Normal Epithelial Cells': np.mean([r['normal_epithelia_proportion'] for r in valid_results]),
            'T cells': np.mean([r['tcells_proportion'] for r in valid_results]),
            'Other Immune Cells': np.mean([r['other_immune_proportion'] for r in valid_results])
        }
        
        print("\nAverage Cell Proportions:")
        for cell_type, proportion in avg_proportions.items():
            print(f"  {cell_type}: {proportion:.3f} ({proportion*100:.1f}%)")
    
    print("="*50)

if __name__ == "__main__":
    main()
