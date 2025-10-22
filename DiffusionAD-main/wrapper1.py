import argparse, os, random, time, json
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader

# import repo functions
from train import train, eval, defaultdict_from_json, load_trained_models
from data.dataset_beta_thresh import (
    MVTecTrainDataset, MVTecTestDataset,
    VisATrainDataset, VisATestDataset,
    DAGMTrainDataset, DAGMTestDataset,
    MPDDTrainDataset, MPDDTestDataset
)
from models.Recon_subnetwork import UNetModel
from models.Seg_subnetwork import SegmentationSubNetwork

# ---------------------------
# Utility helpers
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_label_noise(dataset, noise_level=0.0):
    """Flip a portion of labels to 0 (clean) to simulate annotation noise."""
    if noise_level <= 0:
        return dataset
    n = len(dataset)
    idx = np.random.choice(n, int(n * noise_level), replace=False)
    if hasattr(dataset, "labels"):
        for i in idx:
            dataset.labels[i] = 0
    return dataset


def measure_efficiency(model, device="cuda"):
    dummy_x = torch.randn(1, 3, 256, 256).to(device)
    dummy_t = torch.zeros(1, dtype=torch.long).to(device)  # timestep needed for UNet

    params = sum(p.numel() for p in model.parameters()) / 1e6
    flops = FlopCountAnalysis(model, (dummy_x, dummy_t)).total() / 1e9

    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with torch.no_grad():
        _ = model(dummy_x, dummy_t)   # must pass timestep
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
# Main experiment runner
# ---------------------------
def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    # --- load json config ---
    config_file = f"args{args.config_id}.json"
    with open(os.path.join("args", config_file), "r") as f:
        config = json.load(f)
    config["arg_num"] = args.config_id
    config = defaultdict_from_json(config)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- choose dataset type ---
    if args.dataset == "mvtec":
        root = config["mvtec_root_path"]
        TrainSet, TestSet = MVTecTrainDataset, MVTecTestDataset
        categories = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable',
                      'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw',
                      'toothbrush', 'transistor', 'zipper']
        class_type = "MVTec"
    elif args.dataset == "visa":
        root = config["visa_root_path"]
        TrainSet, TestSet = VisATrainDataset, VisATestDataset
        categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 
                      'macaroni1', 'macaroni2', 'pcb1', 'pcb2','pcb3', 'pcb4', 'pipe_fryum']
        class_type = "VisA"
    elif args.dataset == "mpdd":
        root = config["mpdd_root_path"]
        TrainSet, TestSet = MPDDTrainDataset, MPDDTestDataset
        categories = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
        class_type = "MPDD"
    elif args.dataset == "dagm":
        root = config["dagm_root_path"]
        TrainSet, TestSet = DAGMTrainDataset, DAGMTestDataset
        categories = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
        class_type = "DAGM"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # If user specified a single category, override categories
    if args.category is not None:
        categories = [args.category]

    # ---------------------------
    # Loop over all categories
    # ---------------------------
    # ---------------------------
    # Loop over all categories
    # ---------------------------
    # Load already processed categories from CSV
    import csv

    csv_path = os.path.join(args.output_dir, "results.csv")
    done_categories = []

    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header and "category" in header:
                    cat_idx = header.index("category")
                    for row in reader:
                        if len(row) > cat_idx:
                            done_categories.append(row[cat_idx].strip())
                else:
                    # fallback if header broken or missing
                    for row in reader:
                        if len(row) > 2:
                            done_categories.append(row[2].strip())
            done_categories = list(set(done_categories))  # remove duplicates
        except Exception as e:
            print(f"⚠ Warning: Could not read {csv_path}: {e}")
    else:
        done_categories = []

    print(f"Already completed categories: {done_categories}")


    for sub_class in categories:
        # Skip categories that already have results
        if sub_class in done_categories:
            print(f"→ Skipping {sub_class}, already has results.")
            continue

        print(f"\n=== Running DiffusionAD on {args.dataset}/{sub_class} ===")
        
        subclass_path = os.path.join(root, sub_class)
        training_dataset = TrainSet(subclass_path, sub_class,
                                    img_size=config["img_size"], args=config)
        testing_dataset = TestSet(subclass_path, sub_class,
                                img_size=config["img_size"])

        training_dataset = add_label_noise(training_dataset, args.noise)

        train_loader = DataLoader(training_dataset,
                                batch_size=config["Batch_Size"],
                                shuffle=True, num_workers=1,
                                pin_memory=True, drop_last=True)
        test_loader = DataLoader(testing_dataset,
                                batch_size=1, shuffle=False, num_workers=1)

        # --- train or just eval ---
        if not args.eval_only:
            train(train_loader, test_loader, config,
                len(testing_dataset), sub_class, class_type, device)

        unet_model, seg_model = load_trained_models(config, sub_class, device, which='best')
        image_auroc, pixel_auroc = eval(test_loader, config, unet_model, seg_model, len(testing_dataset), sub_class, device)

        # --- efficiency measurement (mvtec, seed=0, no noise) ---
        efficiency = {}
        if args.noise == 0 and args.seed == 0 and args.dataset == "mvtec":
            dummy_unet = UNetModel(config['img_size'][0], config['base_channels'],
                                channel_mults=config['channel_mults'],
                                dropout=config["dropout"],
                                n_heads=config['num_heads'],
                                n_head_channels=config["num_head_channels"],
                                in_channels=config["channels"]).to(device)
            efficiency = measure_efficiency(dummy_unet, device=device)

        # --- save results per category ---
        os.makedirs(args.output_dir, exist_ok=True)
        result = {
            "method": "DiffusionAD",
            "dataset": args.dataset,
            "category": sub_class,
            "seed": args.seed,
            "noise": args.noise,
            "run_id": run_id,
            "Image-AUROC": image_auroc,
            "Pixel-AUROC": pixel_auroc,
            **efficiency
        }
        df = pd.DataFrame([result])
        df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
        print("Saved →", csv_path)



# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mvtec",
                        help="Dataset to run on (mvtec, visa, mpdd, dagm)")
    parser.add_argument("--category", type=str, default=None,
                        help="Run only on a single category (optional)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="./results/diffusionad")
    parser.add_argument("--config_id", type=int, default=1,
                        help="argsX.json config file to use")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    run_experiment(args)


#Statistical significance (3 seeds)

#python diffusionad_wrapper.py --dataset mvtec --seed 0 --output_dir ./results/diffusionad
#python diffusionad_wrapper.py --dataset mvtec --seed 1 --output_dir ./results/diffusionad
#python diffusionad_wrapper.py --dataset mvtec --seed 2 --output_dir ./results/diffusionad

#Robustness to Noise (10%)

#python diffusionad_wrapper.py --dataset mvtec --seed 0 --noise 0.1 --output_dir ./results/diffusionad

#Domain Shift (MVTec → VisA)

#python diffusionad_wrapper.py --dataset visa --seed 0 --eval_only --output_dir ./results/diffusionad

# DiffusionAD-main/
# │
# ├── train.py
# ├── diffusionad_wrapper.py   ← put it here
# ├── args/
# │   └── args1.json
# ├── datasets/
# │   ├── mvtec/
# │   │   ├── bottle/
# │   │   ├── cable/
# │   │   └── ...
# │   ├── VisA_1class/
# │   ├── dagm/
# │   └── mpdd/
# ├── models/
# │   ├── Recon_subnetwork.py
# │   ├── Seg_subnetwork.py
# │   └── DDPM.py
# └── data/
#     └── dataset_beta_thresh.py
