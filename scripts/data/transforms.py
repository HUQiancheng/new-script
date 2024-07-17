import torch
import numpy as np
from torchvision import transforms

class Resize: # Deprecated since all images are resized to 448x448
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
        depth, R, T, intrinsic_mat = sample['depth'], sample['R'], sample['T'], sample['intrinsic_mat']
        
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
            image = torch.tensor(image, dtype=torch.float32)
        else:
            image = image.clone().detach().float()
        
        if isinstance(label, np.ndarray):
            label = label.astype(np.int32)
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = label.clone().detach().long()
        
        if isinstance(depth, np.ndarray):
            depth = depth.astype(np.float32)
            depth = torch.tensor(depth, dtype=torch.float32)
        else:
            depth = depth.clone().detach().float()
        
        if isinstance(R, np.ndarray):
            R = R.astype(np.float32)
            R = torch.tensor(R, dtype=torch.float32)
        else:
            R = R.clone().detach().float()
        
        if isinstance(T, np.ndarray):
            T = T.astype(np.float32)
            T = torch.tensor(T, dtype=torch.float32)
        else:
            T = T.clone().detach().float()
        
        if isinstance(intrinsic_mat, np.ndarray):
            intrinsic_mat = intrinsic_mat.astype(np.float32)
            intrinsic_mat = torch.tensor(intrinsic_mat, dtype=torch.float32)
        else:
            intrinsic_mat = intrinsic_mat.clone().detach().float()
        
        sample = {
            'image': image,
            'label': label,
            'depth': depth,
            'R': R,
            'T': T,
            'intrinsic_mat': intrinsic_mat,
        }
        
        return sample


class Normalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)
    
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        depth, R, T, intrinsic_mat = sample['depth'], sample['R'], sample['T'], sample['intrinsic_mat']
        image = self.normalize(image)
        sample = {
            'image': image,
            'label': label,
            'depth': depth,
            'R': R,
            'T': T,
            'intrinsic_mat': intrinsic_mat,
        }
        
        return sample
