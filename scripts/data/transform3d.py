import torch
import numpy as np
from torchvision import transforms
class ToTensor:
    def __call__(self, sample):
        coords, labels, scene_id = sample['coords'], sample['labels'], sample['scene_id']
        coords = torch.tensor(coords, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int8)
        # scene_id = torch.tensor(scene_id)



        sample = {
            'coords': coords,
            'labels': labels,
            # 'scene_id': scene_id
        }


        return sample



