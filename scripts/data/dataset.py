import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from scripts.data.transforms import Resize, ToTensor, Normalize

class DSLRDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None):
        self.data_dir = data_dir
        self.split_file = split_file
        self.transform = transform
        self.data_list = self._load_split()
        
    def _load_split(self):
        with open(self.split_file, 'r') as file:
            scene_ids = file.read().splitlines()
        data_list = []
        for scene_id in scene_ids:
            pth_path = os.path.join(self.data_dir, f'{scene_id}.pth')
            if os.path.exists(pth_path):
                data_list.append(pth_path)
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        pth_path = self.data_list[index]
        data = torch.load(pth_path)
        
        original_images = data['original_image']
        semantic_labels = data['2d_semantic_labels']
        
        # Randomly select two image from the .pth file
        img_indices = np.random.choice(len(original_images), size=2, replace=False)
        img_index1, img_index2 = img_indices[0], img_indices[1]
        
        
        original_image = np.stack((original_images[img_index1],original_images[img_index2]))
        semantic_label = np.stack((semantic_labels[img_index1],semantic_labels[img_index2]))
        
        # Convert to CHW format
        original_image = original_image.transpose((0, 3, 1, 2))
        
        sample = {
            'image': original_image,
            'label': semantic_label
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
