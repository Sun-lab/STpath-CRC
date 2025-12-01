"""
STPath-COAD: Feature Extraction using Foundation Models for Colorectal Cancer
============================================================================

This script extracts features from histopathology patches using multiple foundation models
including Conch, ProvGigapath, UNI2h, Virchow, and Virchow2 for colorectal cancer analysis.

Author: Saishi Cui
Date: December 2025

Purpose: Extract and precompute features from colorectal cancer H&E patches using various
foundation models to enable downstream cell type proportion prediction.
"""

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import os
from torchvision import models
from pathlib import Path
from timm.layers import SwiGLUPacked
import re
from conch.open_clip_custom import create_model_from_pretrained




# define the log function
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_Conch_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token)
    return model, transform



# load the pre-trained UNI2h model
def load_UNI2h_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token) 
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
    model_UNI2h = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    return model_UNI2h


# load the pre-trained ProvGigapath model
def load_ProvGigapath_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return tile_encoder


# load the pre-trained Virchow model
def load_Virchow_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow = model_Virchow.eval()
    return model_Virchow


def load_Virchow2_model():
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2 = model_Virchow2.eval()
    return model_Virchow2




class ImageDataset_training(Dataset):
    def __init__(self, img_dir, csv_file, celltype, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.metadata_df = pd.read_csv(csv_file, index_col=0)
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]
        self.img_to_label = dict(zip(self.metadata_df.index, self.metadata_df[celltype]))

    
    def extract_identifier_individual(self, text):
        # Regular expression explanation:
        # .*?-1_  : Match any characters until -1_
        # (.*?)   : Capture group 1 - extract content after -1_
        # (?:_region\d+)?$ : Non-capture group, optional _region+digit ending
        
        pattern = r'.*?-1_(.*?)(?:_region\d+)?$'
        match = re.match(pattern, text)
        
        if match:
            extracted = match.group(1)
            
            # Check if starts with a digit
            if re.match(r'^\d', extracted):
                # Starts with digit, split by _ and take first part
                return extracted.split('_')[0]
            else:
                if extracted == "TENX92" or extracted == "TENX91":
                    extracted = "TENX92"
                elif extracted == "TENX90" or extracted == "TENX89":
                    extracted = "TENX90"
                elif extracted == "ZEN47" or extracted == "ZEN46":
                    extracted = "ZEN47"
                elif extracted == "ZEN45" or extracted == "ZEN44":
                    extracted = "ZEN45"
                return extracted
        
        return None


    def extract_identifier_sample(self, text):
        # Regular expression explanation:
        # .*?-1_  : Match any characters until -1_
        # (.*?)   : Capture group 1 - extract content after -1_
        # (?:_region\d+)?$ : Non-capture group, optional _region+digit ending
        
        pattern = r'.*?-1_(.*?)(?:_region\d+)?$'
        match = re.match(pattern, text)
        
        if match:
            extracted = match.group(1)        
            return extracted
        
        return None




    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        tile_file = self.image_files[idx]
        tile_id = tile_file.split('.')[0]
        
        # Use regex function to extract sample_id and individual_id
        sample_id = self.extract_identifier_sample(tile_id)
        individual_id = self.extract_identifier_individual(tile_id)

        label = self.img_to_label.get(tile_id, 0.0)
        img_path = os.path.join(self.img_dir, tile_file)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32), tile_id, sample_id, individual_id


def create_training_features_Conch(cell_types_list):
    
    batch_size = 32

    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    # load the pre-trained encoder
    model, transform = load_Conch_model()
    model = model.to(device)
    model.eval()


    
    

    # loop through each cell type
    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_Conch.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        
        # create the image dataset
        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)

        
        # create the data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # precompute all features
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)  
                embeddings = model.encode_image(images, proj_contrast=False, normalize=False)
                embeddings_list.append(embeddings.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
        # merge the results of all batches
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
  
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        # save the precomputed features
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        
        log("the precomputation is completed, the features are saved")


def create_training_features_UNI2h(cell_types_list):
    
    batch_size = 32

    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    # load the pre-trained encoder
    tile_encoder = load_UNI2h_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    
    
    # loop through each cell type
    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_UNI2h.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        
        # create the image dataset
        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)

        
        # create the data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # precompute all features
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_list.append(embeddings.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
        # merge the results of all batches
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)

  
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        # save the precomputed features
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        
        log("the precomputation is completed, the features are saved")


def create_training_features_ProvGigapath(cell_types_list):

    batch_size = 32
        # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    # load the pre-trained encoder
    tile_encoder = load_ProvGigapath_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    # image pre-processing
    transform = transforms.Compose(
    [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
    

    # loop through each cell type
    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_ProvGigapath.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue
        
    
        # create the image dataset
        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)  
        # create the data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # precompute all features
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_list.append(embeddings.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
        # merge the results of all batches
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
  
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        # save the precomputed features
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        
        log("the precomputation is completed, the features are saved")


def create_training_features_Virchow(cell_types_list):
    batch_size = 32


    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    # load the pre-trained encoder
    tile_encoder = load_Virchow_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    
    # loop through each cell type
    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_Virchow.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        
        # create the image dataset
        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        print(f"Total dataset size: {len(dataset)}")
        
        # create the data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # precompute all features
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_stacked = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)  
                embeddings_list.append(embeddings_stacked.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
        # merge the results of all batches
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
  
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        # save the precomputed features
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        
        log("the precomputation is completed, the features are saved")


def create_training_features_Virchow2(cell_types_list):
    batch_size = 32


    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    # load the pre-trained encoder
    tile_encoder = load_Virchow2_model()
    tile_encoder = tile_encoder.to(device)
    tile_encoder.eval()
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    
    # loop through each cell type
    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_Virchow2.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue

        
        # create the image dataset
        log("create the image dataset...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        print(f"Total dataset size: {len(dataset)}")
        
        # create the data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # precompute all features
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="calculate the features"):
                images = images.to(device)
                embeddings = tile_encoder(images)
                embeddings_stacked = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)  
                embeddings_list.append(embeddings_stacked.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
        # merge the results of all batches
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)
  
        log(f"the features are calculated, the number of features: {len(embeddings)}")
        
        # save the precomputed features
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        
        log("the precomputation is completed, the features are saved")


def create_training_features_ResNet50(cell_types_list):
    batch_size = 32

    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log("use CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log("use MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        log("use CPU")
    
    # load the pre-trained ResNet50 model
    log("load the ImageNet pre-trained ResNet50 model...")
    model = models.resnet50(pretrained=True)
    
    # remove the last classification layer, only use the feature extraction part
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    # define the ImageNet standard pre-processing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    

    
    for cell_type in cell_types_list:
        print(cell_type)
        img_dir = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_tiles/{cell_type}"
        csv_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_celltype_proportion/{cell_type}_celltype_proportion_combined.csv"
        output_file = f"/Users/scui2/Desktop/Colorectal_Cancer_HE_patches/Training_features/{cell_type}_training_precomputed_features_ResNet50.pt"
        if len(os.listdir(img_dir)) == 0:
            log(f"the image directory for {cell_type} is empty, skip the precomputation")
            continue
        # create the image dataset
        
        log(f"create the image dataset for {cell_type}...")
        dataset = ImageDataset_training(img_dir, csv_file, cell_type, transform)
        log(f"the dataset contains {len(dataset)} images")
        
        # create the data loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # precompute all features
        log(f"start to precompute the features for {len(dataset)} images...")
        embeddings_list = []
        celltype_proportions_list = []
        tile_ids_list = []
        sample_ids_list = []
        individual_ids_list = []
        
        with torch.no_grad():
            for images, celltype_prop, tile_id, sample_id, individual_id in tqdm(loader, desc="Computing ResNet50 features"):
                images = images.to(device)
                
                # feature extraction
                features = feature_extractor(images)
                # the shape of the ResNet50 features is [batch_size, 2048, 1, 1], need to flatten
                features = features.squeeze(-1).squeeze(-1)
                
                embeddings_list.append(features.cpu())
                celltype_proportions_list.append(celltype_prop)
                tile_ids_list.extend(tile_id)
                sample_ids_list.extend(sample_id)
                individual_ids_list.extend(individual_id)
                
        # merge the results of all batches
        embeddings = torch.cat(embeddings_list, 0)
        celltype_proportions = torch.cat(celltype_proportions_list, 0)

  
        log(f"ResNet50 feature calculation completed, the number of features: {len(embeddings)}, the dimension of the features: {embeddings.shape[1]}")
        
        # save the precomputed features
        log(f"save the features to: {output_file}")
        torch.save({
            'embeddings': embeddings,
            'celltype_proportions': celltype_proportions,
            'tile_ids': tile_ids_list,
            'sample_ids': sample_ids_list,
            'individual_ids': individual_ids_list,
        }, output_file)
        
        log(f"{cell_type} ResNet50 feature extraction completed")



if __name__ == "__main__":
    # define the cell types
    cell_types_list = ["Cancer Cells", "Stromal Cells", "Normal Epithelial Cells", "Other Immune Cells", "T Cells"]
    create_training_features_Conch(cell_types_list = cell_types_list)
    create_training_features_ResNet50(cell_types_list = cell_types_list)
    create_training_features_ProvGigapath(cell_types_list = cell_types_list)
    create_training_features_UNI2h(cell_types_list = cell_types_list)
    create_training_features_Virchow(cell_types_list = cell_types_list)
    create_training_features_Virchow2(cell_types_list = cell_types_list)
    




