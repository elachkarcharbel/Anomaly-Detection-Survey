import argparse, os, random, time
import numpy as np

import os

torch_cache_dir = os.path.join(os.getcwd(), "torch_cache")
os.environ['TORCH_HOME'] = torch_cache_dir
os.makedirs(torch_cache_dir, exist_ok=True)

mpl_config_dir = os.path.join(os.getcwd(), "matplotlib_cache")
os.environ['MPLCONFIGDIR'] = mpl_config_dir
os.makedirs(mpl_config_dir, exist_ok=True)

import torch
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from omegaconf import OmegaConf
from datetime import datetime
# import repo modules
#from config import get_config
from train import trainer
from ddad import DDAD
from main import build_model   # reuse their build_model()
import time
import psutil

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_label_noise(dataset, noise_level=0.0):
    if noise_level <= 0:
        return dataset
    n = len(dataset)
    idx = np.random.choice(n, int(n * noise_level), replace=False)
    if hasattr(dataset, "labels"):
        for i in idx:
            dataset.labels[i] = 0
    return dataset

def measure_efficiency(model):


    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, 256, 256).to(device)

    # -------------------------------
    # Count parameters
    # -------------------------------
    try:
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        params_m = "N/A"

    # -------------------------------
    # Compute FLOPs
    # -------------------------------
    try:
        flops_g = FlopCountAnalysis(model, dummy).total() / 1e9
    except Exception:
        flops_g = "N/A"

    # -------------------------------
    # Measure latency
    # -------------------------------
    try:
        start = time.time()
        with torch.no_grad():
            for _ in range(10):  # warm-up runs
                _ = model(dummy)
            torch.cuda.synchronize(device)
            start = time.time()
            for _ in range(30):
                _ = model(dummy)
            torch.cuda.synchronize(device)
        latency = (time.time() - start) / 30 * 1000  # ms/img
    except Exception:
        latency = "N/A"

    # -------------------------------
    # GPU memory usage
    # -------------------------------
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            _ = model(dummy)
            torch.cuda.synchronize(device)
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        else:
            gpu_mem = psutil.virtual_memory().used / (1024 ** 2)
    except Exception:
        gpu_mem = "N/A"

    return {
        "params(M)": params_m,
        "FLOPs(G)": flops_g,
        "latency(ms/img)": latency,
        "gpu_mem(MB)": gpu_mem,
    }



def run_experiment(args):
    set_seed(args.seed)
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = OmegaConf.load(config_path)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    data_root = config.data.data_dir
    categories = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results_0s.csv")


    print("\n=== Measuring model efficiency (once) ===")
    base_model = build_model(config).to(config.model.device)
    base_model.eval()
    efficiency = measure_efficiency(base_model)
    print("✅ Efficiency:", efficiency)

    for category in categories:
        print(f"\n=== Processing category: {category} ===")
        config.data.category = category

        ckpt_dir = os.path.join(config.model.checkpoint_dir, category)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Auto-rename numeric checkpoints to .pth
        for f in os.listdir(ckpt_dir):
            fpath = os.path.join(ckpt_dir, f)
            if os.path.isfile(fpath) and f.isdigit():
                os.rename(fpath, fpath + ".pth")
                print(f" → Auto-renamed {f} → {f}.pth")

        # Find latest checkpoint
        ckpt_steps = [int(f.split(".")[0]) for f in os.listdir(ckpt_dir)
                      if f.split(".")[0].isdigit() and f.endswith(".pth")]
        latest_step = max(ckpt_steps) if ckpt_steps else None

        model = build_model(config).to(config.model.device)
        model = torch.nn.DataParallel(model)

        if latest_step is not None:
            print(f" → Found checkpoint {latest_step}.pth for {category}, skipping training.")
            config.model.load_chp = latest_step
            config.model.DA_chp = latest_step
            ckpt_path = os.path.join(ckpt_dir, f"{latest_step}.pth")
            model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
        else:
            print(f" → No checkpoint found for {category}, training from scratch.")
            trainer(model, category, config)
            final_ckpt = os.path.join(ckpt_dir, "2000.pth")
            torch.save(model.state_dict(), final_ckpt)
            config.model.load_chp = 2000
            config.model.DA_chp = 2000

        # Evaluation
        print(f"Evaluating {category} ...")
        ddad = DDAD(model, config)
        metrics = ddad() or {}

        image_auroc = metrics.get("auroc") or metrics.get("Image-AUROC") or None
        pixel_auroc = metrics.get("pro") or metrics.get("Pixel-AUROC") or None
        pro_metric = metrics.get("pro") or None

        result = {
            "method": "DDAD",
            "dataset": args.dataset,
            "category": category,
            "seed": args.seed,
            "noise": args.noise,
            "run_id": run_id,
            "Image-AUROC": float(image_auroc) if torch.is_tensor(image_auroc) else image_auroc,
            "Pixel-AUROC": float(pixel_auroc) if torch.is_tensor(pixel_auroc) else pixel_auroc,
            "PRO": float(pro_metric) if torch.is_tensor(pro_metric) else pro_metric,
            **efficiency
        }

        df = pd.DataFrame([result])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        print(f"  Saved {category} results → {csv_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./results/ddad")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    run_experiment(args)



# Statistical significance (3 seeds)

#python ddad_wrapper.py --dataset mvtec --seed 0 --output_dir ./results/ddad
#python ddad_wrapper.py --dataset mvtec --seed 1 --output_dir ./results/ddad
#python ddad_wrapper.py --dataset mvtec --seed 2 --output_dir ./results/ddad

# Robustness to Noise (10%)
#python ddad_wrapper.py --dataset mvtec --seed 0 --noise 0.1 --output_dir ./results/ddad

# Domain Shift (MVTec → VisA)
#python ddad_wrapper.py --dataset visa --seed 0 --eval_only --output_dir ./results/ddad
