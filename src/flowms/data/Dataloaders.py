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
            self.h5files = self.h5files[:int(0.1*len(self.h5files))]
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
def train_loader_bccd(size=64, batch_size=8):
    return DataLoader(BCCD(data_raw_dir, size=size, train=True), batch_size=batch_size, shuffle=True)

def test_loader_bccd(size=64, batch_size=8):
    return DataLoader(BCCD(data_raw_dir, size=size, train=False), batch_size=batch_size, shuffle=False)

def train_loader_brats(size=64, batch_size=8):
    return DataLoader(BraTS(data_raw_dir, size=size, train=True), batch_size=batch_size, shuffle=True)

def test_loader_brats(size=64, batch_size=8):
    return DataLoader(BraTS(data_raw_dir, size=size, train=False), batch_size=batch_size, shuffle=False)

