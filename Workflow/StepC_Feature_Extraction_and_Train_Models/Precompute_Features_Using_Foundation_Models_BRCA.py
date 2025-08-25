"""
STPath-COAD: Feature Extraction using Foundation Models for Breast Cancer
========================================================================

This script extracts features from histopathology patches using multiple foundation models
including Conch, ProvGigapath, UNI2h, Virchow, and Virchow2 for breast cancer analysis.

Author: Saishi Cui
Date: Sept 2025


Purpose: Extract features from breast cancer H&E patches using various
foundation models to enable downstream cell type proportion prediction.
"""

import os
import torch
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import os
from torchvision import models
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from conch.open_clip_custom import create_model_from_pretrained


# define the log function
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


# define the image dataset class
# img_dir: the directory of the patches (folder)
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Extract sample_id from filename (remove extension)
        sample_id = img_name.split(".")[0]
        
        return image, sample_id


def load_Conch_model():
    # Replace with your HuggingFace token from https://huggingface.co/settings/tokens
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token)
    return model, transform


# load the pre-trained ProvGigapath model
def load_ProvGigapath_model():
    # Replace with your HuggingFace token from https://huggingface.co/settings/tokens
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    return tile_encoder


# load the pre-trained UNI2h model
def load_UNI2h_model():
    # Replace with your HuggingFace token from https://huggingface.co/settings/tokens
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

# load the pre-trained Virchow model
def load_Virchow_model():
    # Replace with your HuggingFace token from https://huggingface.co/settings/tokens
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow = model_Virchow.eval()
    return model_Virchow


def load_Virchow2_model():
    # Replace with your HuggingFace token from https://huggingface.co/settings/tokens
    hf_token = "YOUR_HUGGINGFACE_TOKEN_HERE"
    login(token=hf_token)
    model_Virchow2 = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model_Virchow2 = model_Virchow2.eval()
    return model_Virchow2


def create_features_Conch(folder_name = "Mo_2024_HTA12_129_7"):
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
    img_dir = f"Breast_Cancer_Patches/{folder_name}"
    output_file = f"Breast_Cancer_Conch_features/{folder_name}_precomputed_features_Conch.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # create the image dataset
    log("create the image dataset...")
    dataset = ImageDataset(img_dir, transform)
                
    # create the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # precompute all features
    log(f"start to precompute the features for {len(dataset)} images...")
    embeddings_list = []
    sample_ids_list = []
    with torch.no_grad():
        for images, sample_ids in tqdm(loader, desc="calculate the features"):
            images = images.to(device)
            embeddings = model.encode_image(images, proj_contrast=False, normalize=False)
            embeddings_list.append(embeddings.cpu())
            sample_ids_list.extend(sample_ids)
    # merge the results of all batches
    embeddings = torch.cat(embeddings_list, 0)
  
    log(f"the features are calculated, the number of features: {len(embeddings)}")
    
    # Create sample_id to features mapping
    features_dict = {}
    for i, sample_id in enumerate(sample_ids_list):
        features_dict[sample_id] = embeddings[i].tolist()
    
    # save the precomputed features
    log(f"save the features to: {output_file}")
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids_list,
        'features_dict': features_dict,  # id_1: [feature1, feature2, ...]
    }, output_file)
    
    log("the precomputation is completed, the features are saved")


def create_features_ResNet50(folder_name = "Mo_2024_HTA12_129_7"):
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

    # loop through each cell type
    img_dir = f"Breast_Cancer_Patches/{folder_name}"
    output_file = f"Breast_Cancer_ResNet50_features/{folder_name}_precomputed_features_ResNet50.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # create the image dataset
    log("create the image dataset...")
    dataset = ImageDataset(img_dir, transform)
                
    # create the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # precompute all features
    log(f"start to precompute the features for {len(dataset)} images...")
    embeddings_list = []
    sample_ids_list = []
    with torch.no_grad():
        for images, sample_ids in tqdm(loader, desc="calculate the features"):
            images = images.to(device)
            features = feature_extractor(images)
            # the shape of the ResNet50 features is [batch_size, 2048, 1, 1], need to flatten
            features = features.squeeze(-1).squeeze(-1)
            embeddings_list.append(features.cpu())
            sample_ids_list.extend(sample_ids)
    # merge the results of all batches
    embeddings = torch.cat(embeddings_list, 0)
  
    log(f"the features are calculated, the number of features: {len(embeddings)}")
    
    # Create sample_id to features mapping
    features_dict = {}
    for i, sample_id in enumerate(sample_ids_list):
        features_dict[sample_id] = embeddings[i].tolist()
    
    # save the precomputed features
    log(f"save the features to: {output_file}")
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids_list,
        'features_dict': features_dict,  # id_1: [feature1, feature2, ...]
    }, output_file)
    
    log("the precomputation is completed, the features are saved")


def create_features_ProvGigapath(folder_name = "Mo_2024_HTA12_129_7"):
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
    
    transform = transforms.Compose([
    transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    # loop through each cell type
    img_dir = f"Breast_Cancer_Patches/{folder_name}"
    output_file = f"Breast_Cancer_ProvGigapath_features/{folder_name}_precomputed_features_ProvGigapath.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # create the image dataset
    log("create the image dataset...")
    dataset = ImageDataset(img_dir, transform)
                
    # create the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # precompute all features
    log(f"start to precompute the features for {len(dataset)} images...")
    embeddings_list = []
    sample_ids_list = []
    with torch.no_grad():
        for images, sample_ids in tqdm(loader, desc="calculate the features"):
            images = images.to(device)
            embeddings = tile_encoder(images)
            embeddings_list.append(embeddings.cpu())
            sample_ids_list.extend(sample_ids)
    # merge the results of all batches
    embeddings = torch.cat(embeddings_list, 0)
  
    log(f"the features are calculated, the number of features: {len(embeddings)}")
    
    # Create sample_id to features mapping
    features_dict = {}
    for i, sample_id in enumerate(sample_ids_list):
        features_dict[sample_id] = embeddings[i].tolist()
    
    # save the precomputed features
    log(f"save the features to: {output_file}")
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids_list,
        'features_dict': features_dict,  # id_1: [feature1, feature2, ...]
    }, output_file)
    
    log("the precomputation is completed, the features are saved")


def create_features_UNI2h(folder_name = "Mo_2024_HTA12_129_7"):
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
    img_dir = f"Breast_Cancer_Patches/{folder_name}"
    output_file = f"Breast_Cancer_UNI2h_features/{folder_name}_precomputed_features_UNI2h.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # create the image dataset
    log("create the image dataset...")
    dataset = ImageDataset(img_dir, transform)
                
    # create the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # precompute all features
    log(f"start to precompute the features for {len(dataset)} images...")
    embeddings_list = []
    sample_ids_list = []
    with torch.no_grad():
        for images, sample_ids in tqdm(loader, desc="calculate the features"):
            images = images.to(device)
            embeddings = tile_encoder(images)
            embeddings_list.append(embeddings.cpu())
            sample_ids_list.extend(sample_ids)
    # merge the results of all batches
    embeddings = torch.cat(embeddings_list, 0)
  
    log(f"the features are calculated, the number of features: {len(embeddings)}")
    
    # Create sample_id to features mapping
    features_dict = {}
    for i, sample_id in enumerate(sample_ids_list):
        features_dict[sample_id] = embeddings[i].tolist()
    
    # save the precomputed features
    log(f"save the features to: {output_file}")
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids_list,
        'features_dict': features_dict,  # id_1: [feature1, feature2, ...]
    }, output_file)
    
    log("the precomputation is completed, the features are saved")


def create_features_Virchow(folder_name = "Mo_2024_HTA12_129_7"):
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
    img_dir = f"Breast_Cancer_Patches/{folder_name}"
    output_file = f"Breast_Cancer_Virchow_features/{folder_name}_precomputed_features_Virchow.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # create the image dataset
    log("create the image dataset...")
    dataset = ImageDataset(img_dir, transform)
                
    # create the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # precompute all features
    log(f"start to precompute the features for {len(dataset)} images...")
    embeddings_list = []
    sample_ids_list = []
    with torch.no_grad():
        for images, sample_ids in tqdm(loader, desc="calculate the features"):
            images = images.to(device)
            embeddings = tile_encoder(images)
            embeddings_stacked = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)  
            embeddings_list.append(embeddings_stacked.cpu())
            sample_ids_list.extend(sample_ids)
    # merge the results of all batches
    embeddings = torch.cat(embeddings_list, 0)
  
    log(f"the features are calculated, the number of features: {len(embeddings)}")
    
    # Create sample_id to features mapping
    features_dict = {}
    for i, sample_id in enumerate(sample_ids_list):
        features_dict[sample_id] = embeddings[i].tolist()
    
    # save the precomputed features
    log(f"save the features to: {output_file}")
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids_list,
        'features_dict': features_dict,  # id_1: [feature1, feature2, ...]
    }, output_file)
    
    log("the precomputation is completed, the features are saved")


def create_features_Virchow2(folder_name = "Mo_2024_HTA12_129_7"):
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
    img_dir = f"Breast_Cancer_Patches/{folder_name}"
    output_file = f"Breast_Cancer_Virchow2_features/{folder_name}_precomputed_features_Virchow2.pt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # create the image dataset
    log("create the image dataset...")
    dataset = ImageDataset(img_dir, transform)
                
    # create the data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    # precompute all features
    log(f"start to precompute the features for {len(dataset)} images...")
    embeddings_list = []
    sample_ids_list = []
    with torch.no_grad():
        for images, sample_ids in tqdm(loader, desc="calculate the features"):
            images = images.to(device)
            embeddings = tile_encoder(images)
            embeddings_stacked = torch.cat([embeddings[:,0], embeddings[:,5:].mean(1)], dim=-1)  
            embeddings_list.append(embeddings_stacked.cpu())
            sample_ids_list.extend(sample_ids)
            
    # merge the results of all batches
    embeddings = torch.cat(embeddings_list, 0)
  
    log(f"the features are calculated, the number of features: {len(embeddings)}")
    
    # Create sample_id to features mapping
    features_dict = {}
    for i, sample_id in enumerate(sample_ids_list):
        features_dict[sample_id] = embeddings[i].tolist()
    
    # save the precomputed features
    log(f"save the features to: {output_file}")
    torch.save({
        'embeddings': embeddings,
        'sample_ids': sample_ids_list,
        'features_dict': features_dict,  # id_1: [feature1, feature2, ...]
    }, output_file)
    
    log("the precomputation is completed, the features are saved")


if __name__ == "__main__":
    # Run all 6 foundation models
    base_dir = "/Users/scui2/ST/Breast_Cancer_Patches"
    folders_name = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder_name in folders_name[50:]:
        # Check if directory exists, skip if not
        img_dir = f"{base_dir}/{folder_name}"
        if not os.path.exists(img_dir):
            log(f"Directory {img_dir} does not exist, skipping...")
            continue
            
        create_features_Conch(folder_name=folder_name)
        create_features_ResNet50(folder_name=folder_name)
        create_features_ProvGigapath(folder_name=folder_name)
        create_features_UNI2h(folder_name=folder_name)
        create_features_Virchow(folder_name=folder_name)
        create_features_Virchow2(folder_name=folder_name)


