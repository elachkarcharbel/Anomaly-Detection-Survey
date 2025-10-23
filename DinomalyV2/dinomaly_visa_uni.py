import torch
import torch.nn as nn
from dataset import get_data_transforms
from dataset import MVTecDataset
from torch.utils.data import DataLoader
from models.uad import ViTill
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from functools import partial
import warnings
import logging
import os

from utils import evaluation_batch

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def build_model(device, pretrained_checkpoint=None):
    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise ValueError("Architecture not in small, base, large.")

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                 attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    )

    if pretrained_checkpoint is not None:
        print(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda(0)
        else:
            map_location = "cpu"

        state_dict = torch.load(pretrained_checkpoint, map_location=map_location)

        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")

    return model


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    import argparse

    parser = argparse.ArgumentParser(description='ViTill Evaluation')
    parser.add_argument('--data_path', type=str, default='../VisA_pytorch')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str,
                        default='vitill_visa_eval')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                        help='Path to pretrained MVTec checkpoint for evaluation')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print_fn(f"Using device: {device}")

    # Build model and load pretrained checkpoint
    model = build_model(device, args.pretrained_checkpoint)

    if args.eval_only:
        batch_size = 16
        for item in item_list:
            test_data = MVTecDataset(
                root=os.path.join(args.data_path, item),
                transform=get_data_transforms(448, 392)[0],
                gt_transform=get_data_transforms(448, 392)[1],
                phase="test"
            )
            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
            results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            print_fn(f"{item}: {results}")
