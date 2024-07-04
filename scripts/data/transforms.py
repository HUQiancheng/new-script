import torch
import numpy as np
from torchvision import transforms

class Resize:
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size)
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)
        if isinstance(label, np.ndarray):
            label = torch.tensor(label)
        image = self.resize(image)
        label = self.resize(label.unsqueeze(0)).squeeze(0)
        return {'image': image, 'label': label}

class ToTensor:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.clone().detach().float()
        
        if isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()
        
        return {'image': image, 'label': label}

class Normalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.normalize(image)
        return {'image': image, 'label': label}
