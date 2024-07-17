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
        depth_images = data['depth_image']
        camera_params = data['camera_params']
        
        # Randomly select an image from the .pth file
        img_index = np.random.randint(len(original_images))
        
        
        original_image = original_images[img_index]
        semantic_label = semantic_labels[img_index]
        depth_image = depth_images[img_index]
        cam_params = camera_params[img_index]
        
        # Convert to CHW format
        original_image = original_image.transpose((2, 0, 1))
        
        sample = {
            'image': original_image,
            'label': semantic_label,
            'depth': depth_image,
            'R': cam_params['R'],
            'T': cam_params['T'],
            'intrinsic_mat': cam_params['intrinsic_mat'],
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
