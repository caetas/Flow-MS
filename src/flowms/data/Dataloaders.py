import torch
from torch.utils.data import DataLoader, Dataset
from config import data_raw_dir, data_dir
from PIL import Image
import os
import numpy as np
from glob import glob
import cv2
from PIL import Image
import h5py
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import Cityscapes

class BCCD(Dataset):
    def __init__(self, root_dir, size=64, train=True):
        self.root_dir = root_dir
        self.size = size
        if train:
            self.images = glob(os.path.join(root_dir, 'BCCD', 'train','original', '*.png'))
        else:
            self.images = glob(os.path.join(root_dir, 'BCCD', 'test','original', '*.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        img = np.array(img)
        mask = Image.open(self.images[idx].replace('original', 'mask')).convert('RGB')
        mask = np.array(mask)
        # crop random 512x512 patch
        x = np.random.randint(0, img.shape[1] - 512)
        y = np.random.randint(0, img.shape[0] - 512)
        img = img[y:y+512, x:x+512]
        mask = mask[y:y+512, x:x+512]
        # resize both to 64x64
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        mask[mask <= 127] = 0
        mask[mask > 127] = 1
        # make mask float3 AND RANGE -1, 1
        #mask = mask.astype(np.float32) / 255.
        #mask = mask * 2 - 1
        img = img.astype(np.float32) / 255.
        img = img * 2 - 1

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        #mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous()
        mask = mask[:,:,0]

        return img, mask

class BraTS(Dataset):
    def __init__(self, root_dir, size=64, train=True):
        self.root_dir = root_dir
        self.size = size
        base_dir = glob(os.path.join(root_dir, 'BraTS2020','*','*','*'))[0]
        # read all the h5 files in the .txt file
        with open(os.path.join(data_dir, 'splits', 'brats_valid.txt'), 'r') as f:
            h5files = f.readlines()
        h5files = [file.strip() for file in h5files]
        self.h5files = [os.path.join(base_dir, file) for file in h5files]
        del h5files
        # get 0.8 of the data for training
        np.random.seed(0)
        np.random.shuffle(self.h5files)
        np.random.seed(None)
        if train:
            self.h5files = self.h5files[:int(0.8*len(self.h5files))]
        else:
            self.h5files = self.h5files[int(0.8*len(self.h5files)):]

    def __len__(self):
        return len(self.h5files)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5files[idx], 'r') as f:
            img = f['image'][()]
            mask = f['mask'][()]
        # resize both to 64x64
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))
        img = img[:,:,:3]
        # normalize each channel
        if img[:,:,0].max() == 0.0 or img[:,:,1].max() == 0.0 or img[:,:,2].max() == 0.0:
            print(self.h5files[idx])
        # print min and max of each channel
        img[:,:,0] = (img[:,:,0] - img[:,:,0].min()) / (img[:,:,0].max() - img[:,:,0].min())
        img[:,:,1] = (img[:,:,1] - img[:,:,1].min()) / (img[:,:,1].max() - img[:,:,1].min())
        img[:,:,2] = (img[:,:,2] - img[:,:,2].min()) / (img[:,:,2].max() - img[:,:,2].min())

        img = img.astype(np.float32)
        img = img * 2 - 1

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        final_mask = np.zeros((self.size, self.size))
        final_mask = mask[:,:,0] + 2*mask[:,:,1] + 3*mask[:,:,2]
        #mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous()

        return img, final_mask
    
class CelebAMaskHQ(Dataset):
    def __init__(self, root_dir, size=64, train=True):
        self.root_dir = root_dir
        self.size = size
        self.imgs = glob(os.path.join(root_dir, 'CelebAMask-HQ', 'imgs', '*'))
        np.random.seed(0)
        np.random.shuffle(self.imgs)
        np.random.seed(None)
        if train:
            self.imgs = self.imgs[:int(0.8*len(self.imgs))]
        else:
            self.imgs = self.imgs[int(0.8*len(self.imgs)):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        img = np.array(img)

        mask = Image.open(self.imgs[idx].replace('imgs', 'masks').replace('.png', '_mask.png')).convert('L')
        mask = np.array(mask)

        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        img = img.astype(np.float32)/255.0
        img = img * 2 - 1
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return img, mask
    
def remap_labels(mask, classes):
    for c in classes:
        if c.train_id == 255 or c.train_id == -1:
            mask[mask == c.id] = 0
        else:
            mask[mask == c.id] = c.train_id
    return mask

class CustomCityscapes(Dataset):
    def __init__(self, root_dir, split='train', mode='fine', target_type='semantic', size=64):
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.target_type = target_type
        dataset = Cityscapes(root_dir, split=split, mode=mode, target_type=target_type)
        self.images = dataset.images
        self.targets = dataset.targets
        self.classes = dataset.classes
        self.size = size
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        og_img = Image.open(self.images[idx]).convert('RGB')
        og_target = Image.open(self.targets[idx][0]).convert('L')
        og_img = np.array(og_img)
        og_target = np.array(target)
        
        og_img = cv2.resize(og_img, (self.size*2, self.size))
        og_target = cv2.resize(og_target, (self.size*2, self.size))
        random_crop = np.random.randint(0, self.size//2)
        img = og_img[:, random_crop:random_crop+self.size].copy()
        target = og_target[:, random_crop:random_crop+self.size].copy()

        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        target = remap_labels(target, self.classes)
        return img, target
    
def train_loader_bccd(size=64, batch_size=8, num_workers=0):
    return DataLoader(BCCD(data_raw_dir, size=size, train=True), batch_size=batch_size, shuffle=True, num_workers=num_workers)

def test_loader_bccd(size=64, batch_size=8):
    return DataLoader(BCCD(data_raw_dir, size=size, train=False), batch_size=batch_size, shuffle=False)

def train_loader_brats(size=64, batch_size=8, num_workers=0):
    return DataLoader(BraTS(data_raw_dir, size=size, train=True), batch_size=batch_size, shuffle=True, num_workers=num_workers)

def test_loader_brats(size=64, batch_size=8):
    return DataLoader(BraTS(data_raw_dir, size=size, train=False), batch_size=batch_size, shuffle=True)

def train_loader_celebamaskhq(size=64, batch_size=8, num_workers=0):
    return DataLoader(CelebAMaskHQ(data_raw_dir, size=size, train=True), batch_size=batch_size, shuffle=True, num_workers=num_workers)

def test_loader_celebamaskhq(size=64, batch_size=8):
    return DataLoader(CelebAMaskHQ(data_raw_dir, size=size, train=False), batch_size=batch_size, shuffle=True)

def train_loader_cityscapes(size=64, batch_size=8, num_workers=0):
    return DataLoader(CustomCityscapes(os.path.join(data_raw_dir, 'Cityscapes'), split='train', mode='fine', target_type='semantic', size=size), batch_size=batch_size, shuffle=True, num_workers=num_workers)

def test_loader_cityscapes(size=64, batch_size=8):
    return DataLoader(CustomCityscapes(os.path.join(data_raw_dir, 'Cityscapes'), split='val', mode='fine', target_type='semantic', size=size), batch_size=batch_size, shuffle=True)
