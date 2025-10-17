import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms

class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=True):
        from glob import glob
        from PIL import Image
        import torch
        import os
        from torchvision import transforms

        self.config = config
        self.is_train = is_train

        self.image_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t*2)-1)
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor()
        ])

        # Gather images
        exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
        self.image_files = []

        if is_train:
            train_dir = os.path.join(root, category, "train", "good")
            for ext in exts:
                self.image_files.extend(glob(os.path.join(train_dir, ext)))
        else:
            test_dir = os.path.join(root, category, "test")
            for subfolder in ["good", "defective"]:
                folder = os.path.join(test_dir, subfolder)
                for ext in exts:
                    self.image_files.extend(glob(os.path.join(folder, ext)))

        print(f"Looking for {'training' if is_train else 'test'} images in: {root}/{category}/{'train/good' if is_train else 'test'}")
        print(f"Found {len(self.image_files)} images.")

    def __getitem__(self, index):
        import os
        from PIL import Image
        import torch

        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)

        # Handle grayscale
        if image.shape[0] == 1:
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)

        if self.is_train:
            label = "good"
            return image, label
        else:
            # Determine label
            label = "good" if "good" in image_file else "defective"

            # Load mask if available
            if self.config.data.mask and label == "defective":
                gt_folder = image_file.replace("/test/defective/", "/ground_truth/defective/").rsplit(".",1)[0] + ".png"
                if os.path.exists(gt_folder):
                    target = Image.open(gt_folder).convert("L")
                    target = self.mask_transform(target)
                else:
                    target = torch.zeros([1, image.shape[1], image.shape[2]])
                    print(f"⚠ Warning: Mask not found for {image_file}, using zero mask.")
            else:
                target = torch.zeros([1, image.shape[1], image.shape[2]])

            return image, target, label

    def __len__(self):
        return len(self.image_files)


