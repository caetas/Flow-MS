import torch
from torch.utils.data import DataLoader, Dataset
from config import data_raw_dir
from PIL import Image
import os
import numpy as np
from glob import glob
import cv2
from PIL import Image

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
        mask[mask > 127] = 255
        mask[mask <= 127] = 0
        # make mask float3 AND RANGE -1, 1
        mask = mask.astype(np.float32) / 255.
        mask = mask * 2 - 1
        img = img.astype(np.float32) / 255.
        img = img * 2 - 1

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous()

        return img, mask
    
def train_loader(size=64, batch_size=8):
    return DataLoader(BCCD(data_raw_dir, size=size, train=True), batch_size=batch_size, shuffle=True)

def test_loader(size=64, batch_size=8):
    return DataLoader(BCCD(data_raw_dir, size=size, train=False), batch_size=batch_size, shuffle=False)