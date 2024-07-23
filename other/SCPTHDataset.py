import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as T
from scripts.data.transforms import Resize, ToTensor, Normalize


class SCPTHDataset(Dataset):
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

        coords = data['vtx_coords']
        # colors = data['sampled_colors']
        labels = data['vtx_labels']
        scene_id = data['scene_id']

        # Convert to torch tensors
        # coords = torch.tensor(coords, dtype=torch.float32)
        # # colors = torch.tensor(colors, dtype=torch.float32)
        # labels = torch.tensor(labels, dtype=torch.int8)
        # # scene_id = torch.tensor(scene_id)

        sample = {
            'coords': coords,
            'labels': labels,
            'scene_id': scene_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
