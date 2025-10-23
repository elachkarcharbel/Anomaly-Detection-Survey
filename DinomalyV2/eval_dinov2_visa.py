import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.uad import ViTill
from models import vit_encoder
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import evaluation_batch
from dataset import get_data_transforms

# --- Custom VisA Dataset ---
class VisADataset(Dataset):
    def __init__(self, root, category, transform=None, gt_transform=None):
        self.transform = transform
        self.gt_transform = gt_transform
        self.root = root
        self.csv_path = os.path.join(root, category, 'image_anno.csv')

        self.samples = []
        df = pd.read_csv(self.csv_path)

        for _, row in df.iterrows():
            img_path = os.path.join(root, row['image'])
            label = 0 if str(row['label']).strip().lower() == 'normal' else 1
            mask_path = None
            if pd.notna(row['mask']):
                mask_path = os.path.join(root, row['mask'])
            self.samples.append((img_path, label, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        mask = torch.zeros((1, crop_size, crop_size))  # default dummy mask
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert('L')
            if self.gt_transform:
                mask = self.gt_transform(mask_img)

        return img, mask, label, img_path


# --- Config ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = 'results/dinomalyv2/checkpoints/cable_seed0_noise0.0/model_final.pth'
visa_root = 'visa'
categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
              'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
image_size = 448
crop_size = 392
batch_size = 16
method_name = 'Dinomaly'
dataset_name = 'visa'
seed = 1
noise = 0.0
run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
csv_path = f'eval_results_{run_id}.csv'

# --- Transforms ---
data_transform, gt_transform = get_data_transforms(image_size, crop_size)

# --- Model Setup ---
encoder = vit_encoder.load('dinov2reg_vit_base_14')
embed_dim, num_heads = 768, 12
target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
decoder = nn.ModuleList([
    VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
             norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttention2)
    for _ in range(8)
])

model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
               target_layers=target_layers, mask_neighbor_size=0,
               fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device)
model.eval()

# --- Evaluation Loop ---
results = []
for category in categories:
    dataset = VisADataset(root=visa_root, category=category, transform=data_transform, gt_transform=gt_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    with torch.no_grad():
        auroc_sp, _, _, auroc_px, _, _, _ = evaluation_batch(model, dataloader, device, max_ratio=0.01, resize_mask=256)

    result = {
        'method': method_name,
        'dataset': dataset_name,
        'category': category,
        'seed': seed,
        'noise': noise,
        'run_id': run_id,
        'Image-AUROC': round(auroc_sp * 100, 2),
        'Pixel-AUROC': round(auroc_px * 100, 2)
    }
    results.append(result)
    print(f"{category}: Image-AUROC={auroc_sp:.4f}, Pixel-AUROC={auroc_px:.4f}")

    # --- Save CSV after each category ---
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved progress to {csv_path}")

