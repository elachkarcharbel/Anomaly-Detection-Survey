from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class DDAD:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            num_workers= config.model.num_workers,
            drop_last=False,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        self.transform = transforms.Compose([
                            transforms.CenterCrop((224)), 
                        ])

    def __call__(self) -> dict:
        # ------------------ Feature extractor ------------------
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()

        # ------------------ Lists to store results ------------------
        labels_list = []
        predictions = []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []

        # ------------------ Evaluation loop ------------------
        with torch.no_grad():
            for input, gt, labels in self.testloader:
                input = input.to(self.config.model.device)
                # Reconstruct image
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                # Compute anomaly map
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)
                
                # Center crop for consistent size
                transform = transforms.Compose([transforms.CenterCrop((224))])
                anomaly_map = transform(anomaly_map)
                gt = transform(gt)

                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)
                gt_list.append(gt)
                reconstructed_list.append(x0)

                # Collect image-level predictions
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(torch.max(pred).item())

        # ------------------ Compute metrics ------------------
        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
        metric.optimal_threshold()

        # Print metrics
        if self.config.metrics.auroc:
            print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
        if self.config.metrics.misclassifications:
            metric.miscalssified()

        # ------------------ Concatenate tensors for visualization ------------------
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)

        if self.config.metrics.visualisation:
            if not os.path.exists('results'):
                os.mkdir('results')
            visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)

        # ------------------ Return dictionary of metrics ------------------
        return {
            "Image-AUROC": metric.image_auroc(),
            "Pixel-AUROC": metric.pixel_auroc()
        }

