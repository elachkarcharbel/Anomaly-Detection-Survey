import argparse, os, random, time, json, subprocess
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from functools import partial
import torch.nn as nn
from dinov1.utils import trunc_normal_

# ---------------------------
# Utility helpers
# ---------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_label_noise(dataset, noise_level=0.0):
    """Randomly flips a portion of labels to 0 (clean) to simulate label noise."""
    if noise_level <= 0:
        return dataset
    n = len(dataset.samples)
    idx = np.random.choice(n, int(n * noise_level), replace=False)
    for i in idx:
        path, _ = dataset.samples[i]
        dataset.samples[i] = (path, 0)
    return dataset


def measure_efficiency(model, device="cuda"):
    dummy_x = torch.randn(1, 3, 448, 448).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    flops = FlopCountAnalysis(model, (dummy_x,)).total() / 1e9

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_x)
    torch.cuda.synchronize()

    latency = (time.time() - start) * 1000
    mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "params(M)": params,
        "FLOPs(G)": flops,
        "latency(ms/img)": latency,
        "gpu_mem(MB)": mem
    }


# ---------------------------
# Main Experiment Runner
# ---------------------------

def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- dataset setup ---
    if args.dataset == "mvtec":
        data_path = "../mvtec_anomaly_detection"
        categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    elif args.dataset == "visa":
        data_path = "../VisA_pytorch"
        categories = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # --- if a specific category is given ---
    if args.category is not None:
        categories = [args.category]

    # --- prepare CSV ---
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results_v.csv")
    if os.path.exists(csv_path):
        done_df = pd.read_csv(csv_path)
        done_keys = done_df[["dataset", "category", "seed", "noise"]].astype(str).agg("_".join, axis=1).tolist()
    else:
        done_keys = []

    # --- run loop ---
    for sub_class in categories:
        key = f"{args.dataset}_{sub_class}_{args.seed}_{args.noise}"
        if key in done_keys:
            print(f"→ Skipping {sub_class}, already in results.csv")
            continue

        print(f"\n=== Running DinomalyV2 on {args.dataset}/{sub_class} ===")

        # =====================
        # TRAIN / EVAL STEP
        # =====================
        if args.eval_only:
            # Domain shift mode (e.g., MVTec → VisA)
            script = "dinomaly_visa_uni.py"
        else:
            # Normal training mode
            script = "dinomaly_mvtec_uni.py"

        cmd = [
            "python", script,
            "--data_path", data_path,
            "--save_dir", os.path.join(args.output_dir, "checkpoints"),
            "--save_name", f"{sub_class}_seed{args.seed}_noise{args.noise}"
        ]

        if args.eval_only:
            pretrained_ckpt = "results/dinomalyv2/checkpoints/cable_seed0_noise0.0/model_final.pth"
            cmd.append("--pretrained_checkpoint")
            cmd.append(pretrained_ckpt)


        # Inject noise (handled internally by dataset if supported)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # =====================
        # Parse metrics
        # =====================
        log_file = os.path.join(args.output_dir, "checkpoints", f"{sub_class}_seed{args.seed}_noise{args.noise}", "log.txt")
        if not os.path.exists(log_file):
            print(f"⚠️ Warning: log file not found for {sub_class}")
            continue

        image_auroc = pixel_auroc = pro = np.nan
        with open(log_file, "r") as f:
            for line in f:
                if "Mean:" in line:
                    parts = line.strip().split(",")
                    try:
                        image_auroc = float(parts[0].split(":")[-1])
                        pixel_auroc = float(parts[3].split(":")[-1])
                        pro = float(parts[-1].split(":")[-1])
                    except Exception:
                        pass

        # =====================
        # Measure efficiency (only once)
        # =====================
        efficiency = {}
        if args.dataset == "mvtec" and args.seed == 0 and args.noise == 0:
            encoder = vit_encoder.load("dinov2reg_vit_base_14")
            embed_dim, num_heads = 768, 12
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

            bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])
            decoder = nn.ModuleList([
                VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                         attn=LinearAttention2)
                for _ in range(8)
            ])

            model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                           target_layers=target_layers, mask_neighbor_size=0,
                           fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder).to(device)
            efficiency = measure_efficiency(model, device=device)

        # =====================
        # Save result to CSV
        # =====================
        result = {
            "method": "DinomalyV2",
            "dataset": args.dataset,
            "category": sub_class,
            "seed": args.seed,
            "noise": args.noise,
            "run_id": run_id,
            "Image-AUROC": image_auroc,
            "Pixel-AUROC": pixel_auroc,
            "PRO": pro,
            **efficiency
        }

        df = pd.DataFrame([result])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        print(f"✅ Saved results for {sub_class} → {csv_path}")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./results/dinomalyv2")
    args = parser.parse_args()
    run_experiment(args)



# DinomalyV2/
# │
# ├── dinomaly_mvtec_uni.py
# ├── dinomaly_visa_uni.py
# ├── dinomalyv2_wrapper.py   ← put this file here
# ├── models/
# │   ├── uad.py
# │   ├── vit_encoder.py
# │   └── vision_transformer.py
# └── results/
#     └── dinomalyv2/
#         └── results.csv



# Statistical Significance (3 Seeds)

# python dinomalyv2_wrapper.py --dataset mvtec --seed 0 --output_dir ./results/dinomalyv2
# python dinomalyv2_wrapper.py --dataset mvtec --seed 1 --output_dir ./results/dinomalyv2
# python dinomalyv2_wrapper.py --dataset mvtec --seed 2 --output_dir ./results/dinomalyv2

# Robustness to Noise (10%)

# python dinomalyv2_wrapper.py --dataset mvtec --seed 0 --noise 0.1 --output_dir ./results/dinomalyv2

# Domain Shift (MVTec → VisA)

# python dinomalyv2_wrapper.py --dataset visa --seed 0 --eval_only --output_dir ./results/dinomalyv2
