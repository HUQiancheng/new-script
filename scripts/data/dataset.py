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
        
        original_image = data['original_image']
        semantic_label = data['2d_semantic_labels']
        scene_id = data['scene_id']
        # depth_image = data['depth_image']
        # camera_params = data['camera_params']
        
        # Randomly select two image from the .pth file
        img_indices = np.random.choice(len(original_image), size=2, replace=False)
        img_index1, img_index2 = img_indices[0], img_indices[1]
    
        original_images = np.stack((original_image[img_index1],original_image[img_index2]))
        semantic_labels = np.stack((semantic_label[img_index1],semantic_label[img_index2]))
        # depth_images = np.stack((depth_image[img_index1],depth_image[img_index2]))
        # cam_params_R = np.stack((camera_params[img_index1]['R'],camera_params[img_index2]['R']))
        # cam_params_T = np.stack((camera_params[img_index1]['T'],camera_params[img_index2]['T']))
        # cam_params_intrinsic = np.stack((camera_params[img_index1]['intrinsic_mat'],camera_params[img_index2]['intrinsic_mat']))
        
        # Convert to CHW format
        original_images = original_images.transpose((0, 3, 1, 2))

        sample = {
            'image': original_images,
            'label': semantic_labels,
        }
        
        if self.transform:
            sample = self.transform(sample)

        # sample['depth'] = depth_images
        sample['scene_id'] = scene_id
        # sample['R'] = cam_params_R
        # sample['T'] = cam_params_T
        # sample['intrinsic_mat'] = cam_params_intrinsic
        
        return sample
