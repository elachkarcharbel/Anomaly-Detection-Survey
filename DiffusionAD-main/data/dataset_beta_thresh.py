import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random
from data.perlin import rand_perlin_2d_np


texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']

class MVTecTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample
    

class MVTecTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):

        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_path, "*", "*.jpg")))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        

        #foreground path of textural classes
        foreground_path = os.path.join(args["mvtec_root_path"],'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

        

    
    def __len__(self):
        return len(self.image_paths)

    def random_choice_foreground_path(self):
        foreground_path_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_path_id]
        return foreground_path


    def get_foreground_mvtec(self,image_path):
        classname = self.classname
        if classname in texture_list:
            foreground_path = self.random_choice_foreground_path()
        else:
            foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path



    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  
                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)  

                msk = (object_perlin).astype(np.float32) 
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
                
            
            if self.classname in texture_list: # only DTD
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else: # DTD and self-augmentation
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  # >0.5 is DTD 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else: #self-augmentation
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2
            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground_mvtec(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0 
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class VisATestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, category, img_size=(256,256)):
        self.img_size = img_size
        self.category = category

        self.normal_dir = os.path.join(data_path,"Data", "Images", "Normal")
        self.anomaly_dir = os.path.join(data_path,"Data", "Images", "Anomaly")
        self.mask_dir = os.path.join(data_path,"Data", "Masks", "Anomaly")  # may not exist

        self.normal_images = sorted(os.listdir(self.normal_dir))
        self.anomaly_images = sorted(os.listdir(self.anomaly_dir))
        self.all_images = self.normal_images + self.anomaly_images
        self.labels = [0]*len(self.normal_images) + [1]*len(self.anomaly_images)

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

        # Build a mapping of mask filenames for quick lookup
        self.mask_files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(self.mask_dir)}

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        label = self.labels[idx]

        if label == 0:
            img_path = os.path.join(self.normal_dir, img_name)
            mask = torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.float32)
        else:
            img_path = os.path.join(self.anomaly_dir, img_name)
            # Lookup mask by filename without extension (case insensitive)
            base_name = os.path.splitext(img_name)[0].lower()
            mask_filename = self.mask_files.get(base_name, None)
            if mask_filename:
                mask_path = os.path.join(self.mask_dir, mask_filename)
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize(self.img_size)
                mask = torch.tensor(np.array(mask), dtype=torch.float32) / 255.0
                mask = (mask > 0.5).float().unsqueeze(0)
            else:
                mask = torch.zeros(1, self.img_size[0], self.img_size[1], dtype=torch.float32)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "mask": mask,
            "has_anomaly": torch.tensor(label, dtype=torch.float32)
        }



class VisATrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]
        self.image_paths = sorted(glob.glob(self.root_dir+"/*.JPG"))
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_path, "*", "*.jpg")))
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
    
    def __len__(self):
        return len(self.image_paths)


    def get_foreground(self,image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path 


    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  

            min_perlin_scale = 0


            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(
                    np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        # Randomly sample an index
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]

        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Missing train image: {image_path}, skipping.")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # try next index

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image = image.copy()

        # Load corresponding thresh / foreground image
        thresh_path = self.get_foreground(image_path)
        thresh = cv2.imread(thresh_path, 0)
        if thresh is None:
            print(f"[WARN] Missing thresh image: {thresh_path}, using blank mask.")
            thresh = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.uint8)
        else:
            thresh = cv2.resize(thresh, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # Normalize
        thresh = thresh.astype(np.float32) / 255.0
        image = image.astype(np.float32) / 255.0

        # Randomly pick an anomaly source image
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        if not os.path.exists(anomaly_path):
            print(f"[WARN] Missing anomaly image: {anomaly_path}, skipping.")
            return self.__getitem__((idx + 1) % len(self.image_paths))  # try next index

        # Generate augmented image and anomaly mask
        augmented_image, anomaly_mask, has_anomaly = self.perlin_synthetic(
            image, thresh, anomaly_path, cv2_image, thresh_path
        )

        # Transpose to C,H,W
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

        sample = {
            'image': image,
            'anomaly_mask': anomaly_mask,
            'augmented_image': augmented_image,
            'has_anomaly': has_anomaly,
            'idx': idx
        }

        return sample



class DAGMTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.PNG"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_label.PNG"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class DAGMTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.PNG"))
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_path, "*", "*.jpg")))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        #foreground path of textural classes
        foreground_path = os.path.join(args["mvtec_root_path"],'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

        

    
    def __len__(self):
        return len(self.image_paths)

    def random_choice_foreground(self):
        foreground_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_id]
        return foreground_path

    def get_foreground(self,image_path):
        foreground_path = self.random_choice_foreground()
        return foreground_path 

    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6
            min_perlin_scale = 0


            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50): 

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(
                    np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0:
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)

                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0


            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  
        image = np.array(image).astype(np.float32)/255.0

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class MPDDTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class MPDDTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(os.path.join(self.anomaly_source_path, "*", "*.jpg")))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        
        

    
    def __len__(self):
        return len(self.image_paths)

    def get_foreground(self,image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path 


    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50): 

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5: 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate((np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0


            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  # [0,255]->[0,1]
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

