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
        scene_id = data['scene_id']

        
        # Randomly select two image from the .pth file
        img_indices = np.random.choice(len(original_images), size=2, replace=False)
        img_indices.astype(np.longlong)
        img_index1, img_index2 = img_indices[0], img_indices[1]        
        
        original_image = np.stack((original_images[img_index1],original_images[img_index2]))
        semantic_label = np.stack((semantic_labels[img_index1],semantic_labels[img_index2]))
        depth_image = np.stack((depth_images[img_index1],depth_images[img_index2]))
        # Convert camera parameters to numpy arrays and stack
        R = np.stack((camera_params[img_index1]['R'].numpy(), camera_params[img_index2]['R'].numpy()))
        T = np.stack((camera_params[img_index1]['T'].numpy(), camera_params[img_index2]['T'].numpy()))
        intrinsic_mat = np.stack((camera_params[img_index1]['intrinsic_mat'], camera_params[img_index2]['intrinsic_mat']))        
        
        # Convert to CHW format
        original_image = original_image.transpose((0, 3, 1, 2))
        
        sample = {
            'image': original_image,
            'label': semantic_label,
            'depth': depth_image,
            'R': R,
            'T': T,
            'intrinsic_mat': intrinsic_mat,
            'scene_id': scene_id  # TODO: Add scene_id to the sample
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample