import os
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from omegaconf import OmegaConf
from train import trainer
from ddad import DDAD
from main import build_model

# -------------------------
# Environment setup
# -------------------------
torch_cache_dir = os.path.join(os.getcwd(), "torch_cache")
os.environ['TORCH_HOME'] = torch_cache_dir
os.makedirs(torch_cache_dir, exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def measure_efficiency(model):
    import time
    import psutil
    from fvcore.nn import FlopCountAnalysis

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(1, 3, 256, 256).to(device)

    # Params
    try:
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        params_m = "N/A"
    # FLOPs
    try:
        flops_g = FlopCountAnalysis(model, dummy).total() / 1e9
    except Exception:
        flops_g = "N/A"
    # Latency
    try:
        start = time.time()
        with torch.no_grad():
            for _ in range(10): _ = model(dummy)
            torch.cuda.synchronize(device)
            start = time.time()
            for _ in range(30): _ = model(dummy)
            torch.cuda.synchronize(device)
        latency = (time.time() - start)/30*1000
    except Exception:
        latency = "N/A"
    # GPU memory
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            _ = model(dummy)
            torch.cuda.synchronize(device)
            gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
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

# -------------------------
# Zero-shot evaluation
# -------------------------
def run_experiment(args):
    set_seed(args.seed)
    config = OmegaConf.load("config.yaml")

    # Set dataset root
    data_root = "datasets/visa"
    visa_categories = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results_visa_zero_shot.csv")

    # Build model and measure efficiency
    print("\n=== Preparing model for efficiency measurement ===")
    model = build_model(config).to(config.model.device)
    efficiency = measure_efficiency(model)
    print("✅ Efficiency:", efficiency)

    # Load a single MVTec checkpoint
    from collections import OrderedDict
    mvtec_ckpt = "checkpoints/MVTec/cable/2000.pth"
    print(f"\n=== Loading MVTec checkpoint: {mvtec_ckpt} ===")
    state_dict = torch.load(mvtec_ckpt, map_location=config.model.device)
    

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")  # remove `module.` prefix
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model)

    for category in visa_categories:
        print(f"\n=== Zero-shot evaluation on VisA category: {category} ===")
        # Patch dataset paths
        train_dir = os.path.join(data_root, category, "Data", "Images", "Normal")
        test_dir = os.path.join(data_root, category, "Data", "Images", "Anomaly")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print(f"⚠ Skipping {category}, missing data folder.")
            continue

        # Override config paths
        config.data.category = category
        config.data.data_dir = data_root

        # Evaluation
        ddad = DDAD(model, config)
        metrics = ddad() or {}

        import numpy as np
        from sklearn.metrics import roc_auc_score

        def safe_auroc(y_true, y_score):
            try:
                if len(np.unique(y_true)) < 2:
                    return float('nan')
                return roc_auc_score(y_true, y_score)
            except:
                return float('nan')

        # Image-level AUROC
        image_targets = []
        image_scores = []
        for img_score, label in zip(metrics.get("image_scores", []), metrics.get("labels", [])):
            image_scores.append(img_score)
            image_targets.append(0 if label=="good" else 1)

        image_auroc = safe_auroc(np.array(image_targets), np.array(image_scores))

        # Pixel-level AUROC
        pixel_targets = []
        pixel_scores = []
        if "masks" in metrics and "preds" in metrics:
            for mask, pred in zip(metrics["masks"], metrics["preds"]):
                mask_flat = mask.flatten()
                if mask_flat.sum() == 0:
                    continue
                pixel_targets.extend(mask_flat.tolist())
                pixel_scores.extend(pred.flatten().tolist())

        pixel_auroc = safe_auroc(np.array(pixel_targets), np.array(pixel_scores))


        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        result = {
            "method": "DDAD",
            "dataset": "visa",
            "category": category,
            "seed": args.seed,
            "noise": args.noise,
            "run_id": run_id,
            "Image-AUROC": float(image_auroc) if torch.is_tensor(image_auroc) else image_auroc,
            "Pixel-AUROC": float(pixel_auroc) if torch.is_tensor(pixel_auroc) else pixel_auroc,
            **efficiency
        }

        df = pd.DataFrame([result])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        print(f"  Saved {category} results → {csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./results/ddad_visa")
    args = parser.parse_args()
    run_experiment(args)
