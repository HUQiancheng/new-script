import torch
import torch.nn as nn
import spconv.pytorch as spconv
import numpy as np
from spconv.pytorch.utils import PointToVoxel

# 数据加载函数
def load_pth(file_path):
    return torch.load(file_path)

# Voxelizer 类
class Voxelizer:
    def __init__(self, device):
        self.device = device
        self.point_to_voxel_converter = PointToVoxel(
            vsize_xyz=[0.05, 0.05, 0.05], 
            coors_range_xyz=[-1, -2, -2, 1, 2, 2], 
            num_point_features=101,
            max_num_voxels=2000, 
            max_num_points_per_voxel=25,
            device=device
        )
        
    def generate_voxels(self, pc):
        voxels, coords, num_points_per_voxel = self.point_to_voxel_converter(pc, empty_mean=True)
        return voxels, coords, num_points_per_voxel

# VoxelEncoder 类
class VoxelEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, voxels, num_points_per_voxel):
        points_mean = voxels.sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_points_per_voxel.view(-1, 1), min=1.0).type_as(voxels)
        points_mean = points_mean / normalizer
        return points_mean

# TensorHelper 类
class TensorHelper:
    @staticmethod
    def create_spconv_tensor(encoded_features, voxel_coords, batch_size, spatial_shape):
        all_features = []
        all_coords = []

        for i in range(batch_size):
            features = encoded_features[i]
            coords = voxel_coords[i]
            batch_indices = torch.full((features.shape[0], 1), i, dtype=torch.int32).to(features.device)
            coords = torch.cat((batch_indices, coords), dim=1)
            all_features.append(features)
            all_coords.append(coords)
        
        all_features = torch.cat(all_features, dim=0)
        all_coords = torch.cat(all_coords, dim=0)
        
        spconv_tensor = spconv.SparseConvTensor(
            all_features, all_coords, spatial_shape, batch_size
        )
        return spconv_tensor

# SimpleSpConvNet 类
class SimpleSpConvNet(nn.Module):
    def __init__(self, input_channels, spatial_shape):
        super().__init__()
        self.conv_input = spconv.SparseSequential(
            spconv.SparseConv3d(input_channels, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 32, 3, padding=1),
        )
        
    def forward(self, x):
        x = self.conv_input(x)
        return x

# main 函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_file = "/home/lukas/Desktop/new-script/other/try_3d/point_cloud_features_epoch_1000_batch_0.pth"
    point_cloud_features = load_pth(pth_file)
    
    print('Tensor shape', point_cloud_features.shape)
    print('Tensor type', point_cloud_features.dtype)
    
    batch_size = point_cloud_features.shape[0]
    voxelizer = Voxelizer(device)
    voxel_encoder = VoxelEncoder()
    
    all_voxels = []
    all_coords = []
    all_num_points = []

    for i in range(batch_size):
        pc = point_cloud_features[i].to(device)
        voxels, coords, num_points_per_voxel = voxelizer.generate_voxels(pc)
        encoded_features = voxel_encoder(voxels, num_points_per_voxel)
        
        all_voxels.append(encoded_features)
        all_coords.append(coords)
        all_num_points.append(num_points_per_voxel)
    
    spatial_shape = [40, 80, 80]
    spconv_tensor = TensorHelper.create_spconv_tensor(all_voxels, all_coords, batch_size, spatial_shape)
    
    input_channels = all_voxels[0].shape[1]
    model = SimpleSpConvNet(input_channels, spatial_shape)
    model.to(device)
    output = model(spconv_tensor)
    
    print('Output shape:', output.features.shape)

if __name__ == "__main__":
    main()
