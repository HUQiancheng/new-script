import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from scripts.data.transforms import Resize, ToTensor, Normalize

class DSLRDataset(Dataset):
    def __init__(self, data_dir, split_file, transform=None):
        self.data_dir_2d = os.path.join(data_dir, 'data_2d')
        self.data_dir_3d = os.path.join(data_dir, 'data_3d')
        self.split_file = split_file
        self.transform = transform
        self.data_list = self._load_split()
        
    def _load_split(self):
        with open(self.split_file, 'r') as file:
            scene_ids = file.read().splitlines()
        data_list = []
        for scene_id in scene_ids:
            pth_path_2d = os.path.join(self.data_dir_2d, f'{scene_id}.pth')
            pth_path_3d = os.path.join(self.data_dir_3d, f'{scene_id}.pth')
            if os.path.exists(pth_path_2d) and os.path.exists(pth_path_3d):
                data_list.append((pth_path_2d, pth_path_3d))
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        pth_path_2d, pth_path_3d = self.data_list[index]
        

        data_2d = torch.load(pth_path_2d)
        data_3d = torch.load(pth_path_3d)
        
        original_images = data_2d['original_image']
        semantic_labels = data_2d['2d_semantic_labels']
        depth_images = data_2d['depth_image']
        camera_params = data_2d['camera_params']
        
        coords = data_3d['vtx_coords']
        labels = data_3d['vtx_labels']
        
        # Randomly select two images from the .pth file
        img_indices = np.random.choice(len(original_images), size=2, replace=False)
        img_index1, img_index2 = img_indices[0], img_indices[1]
        
        original_image = np.stack((original_images[img_index1], original_images[img_index2]))
        semantic_label = np.stack((semantic_labels[img_index1], semantic_labels[img_index2]))
        depth_image = np.stack((depth_images[img_index1], depth_images[img_index2]))
        
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

            'coord_pc': coords,
            'label_pc': labels

        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
