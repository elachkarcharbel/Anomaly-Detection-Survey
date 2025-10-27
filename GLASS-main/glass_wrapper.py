#!/usr/bin/env python3
import argparse, os, random, time, json, subprocess
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
import backbones, glass
import torch.nn as nn


# -----------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measure_efficiency(model: nn.Module, device="cuda"):
    """Measure params, FLOPs, latency, and GPU memory usage for backbone."""
    dummy_x = torch.randn(1, 3, 288, 288).to(device)

    # use only the backbone for FLOP analysis
    backbone = model.backbone if hasattr(model, "backbone") else model
    params = sum(p.numel() for p in backbone.parameters()) / 1e6

    try:
        flops = FlopCountAnalysis(backbone, (dummy_x,)).total() / 1e9
    except Exception as e:
        print(f"⚠️ FLOPs estimation failed: {e}")
        flops = float("nan")

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        _ = backbone(dummy_x)
    torch.cuda.synchronize()
    latency = (time.time() - start) * 1000
    mem = torch.cuda.max_memory_allocated() / 1024**2

    return {
        "params(M)": round(params, 2),
        "FLOPs(G)": round(flops, 2) if not np.isnan(flops) else None,
        "latency(ms/img)": round(latency, 2),
        "gpu_mem(MB)": round(mem, 2),
    }



def read_metrics_from_csv(csv_path):
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        if "Mean" in df.iloc[-1, 0]:
            df = df.iloc[:-1]
        last = df.iloc[-1]
        return {
            "Image-AUROC": float(last.get("image_auroc", np.nan)),
            "Image-AP": float(last.get("image_ap", np.nan)),
            "Pixel-AUROC": float(last.get("pixel_auroc", np.nan)),
            "Pixel-AP": float(last.get("pixel_ap", np.nan)),
            "PRO": float(last.get("pixel_pro", np.nan)),
            "best_epoch": int(last.get("best_epoch", -1)),
        }
    except Exception:
        return {}


# -----------------------------------------------------------
# Main Experiment Runner
# -----------------------------------------------------------

def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- dataset setup ---
    if args.dataset == "mvtec":
        data_path = "../mvtec_anomaly_detection"
        aug_path = "../dtd/images"
        categories = sorted(
            [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        )
    elif args.dataset == "visa":
        data_path = "../visa"
        aug_path = "../dtd/images"
        categories = sorted(
            [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.category:
        categories = [args.category]

    # --- prepare output ---
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    if os.path.exists(csv_path):
        done_df = pd.read_csv(csv_path)
        done_keys = done_df[["dataset", "category", "seed", "noise"]].astype(str).agg("_".join, axis=1).tolist()
    else:
        done_keys = []

    # --- main loop ---
    for cat in categories:
        key = f"{args.dataset}_{cat}_{args.seed}_{args.noise}"
        if key in done_keys:
            print(f"→ Skipping {cat}, already done")
            continue

        print(f"\n=== Running GLASS on {args.dataset}/{cat} (seed={args.seed}, noise={args.noise}) ===")
        ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"{cat}_seed{args.seed}_noise{args.noise}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # training or eval
        cmd = [
            "python", "main.py",
            "--gpu", "0",
            "--seed", str(args.seed),
            "--test", "ckpt" if not args.eval_only else "test",
            "net",
            "-b", "wideresnet50",
            "-le", "layer2", "-le", "layer3",
            "--pretrain_embed_dimension", "1536",
            "--target_embed_dimension", "1536",
            "--patchsize", "3",
            "--meta_epochs", str(args.epochs),
            "--eval_epochs", "1",
            "--dsc_layers", "2",
            "--dsc_hidden", "1024",
            "--pre_proj", "1",
            "--mining", "1",
            "--noise", str(args.noise),
            "--radius", "0.75",
            "--p", "0.5",
            "--step", "20",
            "--limit", "392",
            "--lr", str(args.lr),
            "dataset",
            "--distribution", "0",
            "--mean", "0.5",
            "--std", "0.1",
            "--fg", "0",
            "--rand_aug", "1",
            "--batch_size", str(args.batch_size),
            "--resize", str(args.img_size),
            "--imagesize", str(args.img_size),
            "-d", cat,
            args.dataset, data_path, aug_path
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        # --- parse metrics ---
        result_csv = os.path.join("results", "results.csv")
        if not os.path.exists(result_csv):
            result_csv = os.path.join("results", "eval", f"{cat}", "results.csv")
        metrics = read_metrics_from_csv(result_csv)
        if not metrics:
            print(f"⚠️ No metrics found for {cat}")
            continue

        # --- measure efficiency (once only) ---
        efficiency = {}
        if args.dataset == "mvtec" and args.seed == 0 and args.noise == 0:
            backbone = backbones.load("wideresnet50")
            model = glass.GLASS(device)
            model.load(
                backbone=backbone,
                layers_to_extract_from=["layer2", "layer3"],
                device=device,
                input_shape=(3, args.img_size, args.img_size),
                pretrain_embed_dimension=1536,
                target_embed_dimension=1536,
                meta_epochs=args.epochs,
                lr=args.lr,
            )
            efficiency = measure_efficiency(model, device=device)

        # --- save results to global CSV ---
        result = {
            "method": "GLASS",
            "dataset": args.dataset,
            "category": cat,
            "seed": args.seed,
            "noise": args.noise,
            "run_id": run_id,
            **metrics,
            **efficiency,
        }
        df = pd.DataFrame([result])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        print(f"✅ Saved results for {cat} → {csv_path}")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec", help="mvtec or visa")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=288)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./results/glass")
    args = parser.parse_args()
    run_experiment(args)



#python glass_wrapper.py --dataset mvtec --seed 0 --output_dir ./results/glass
#python glass_wrapper.py --dataset mvtec --seed 1 --output_dir ./results/glass
#python glass_wrapper.py --dataset mvtec --seed 2 --output_dir ./results/glass


#python glass_wrapper.py --dataset mvtec --seed 0 --noise 0.1 --output_dir ./results/glass


#python glass_wrapper.py --dataset visa --seed 0 --eval_only --output_dir ./results/glass
