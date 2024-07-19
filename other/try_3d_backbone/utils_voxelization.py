import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel

"""
====================================
点云特征到稀疏卷积张量的转换流程
====================================

每个类都有明确的职责，处理数据并将结果传递给下一个类，从而实现一个端到端的点云特征处理管道。主要类包括：

1. Voxel 相关
-------------
    - Voxelizer 类: 
        - 负责将点云数据转换为体素数据。
        - 包含 generate_voxels 方法，从点云数据生成体素特征、体素坐标和每个体素中的点数。
    - VoxelEncoder 类:
        - 负责编码体素特征。
        - 包含 forward 方法，处理批量点云数据并进行体素化和编码。
        - 包含 encode_voxels 方法，通过求平均值编码体素特征。

2. Tensor 相关
--------------
    - TensorHelper 类:
        - 负责将编码后的特征和坐标转换为 Spconv 所需的稀疏张量。
        - 包含 create_spconv_tensor 方法，合并批次中的特征和坐标并创建稀疏张量。
    - PC2Tensor 类（端到端封装）:
        - 负责将点云特征输入转换为 Spconv 稀疏张量输出。
        - 包含 forward 方法，调用 VoxelEncoder 进行体素化和编码，并通过 TensorHelper 创建稀疏张量。

未来开发方向
=============
- 目前的 Voxelizer 只是简单地将点云数据转换为体素。Spatial Shape并没有涉及, 可能导致转化失败。
- VoxelEncoder 采用的是求平均值的方式进行编码。但是label信息并没有被考虑进去, 这是整数, 需要采取最频繁的点。
- 考虑结合Kaggle那篇文章, 设计dataset, 最重要的3D几何信息是要与2D匹配。
"""



#########################   数据相关    #####################
# 数据加载函数（网络中不需要）
def load_pth(file_path):
    return torch.load(file_path)


#########################  Voxel相关   ######################
class Voxelizer:
    def __init__(self, device):
        """
        初始化 Voxelizer。

        参数:
            device (torch.device): 运行计算的设备。
        """
        self.device = device
        self.point_to_voxel_converter = PointToVoxel(
            vsize_xyz=[0.1, 0.1, 0.1],  #[0.05, 0.05, 0.05]还是内存不够
            coors_range_xyz=[-1, -1, -1, 2, 3, 3], 
            num_point_features=101,
            max_num_voxels=2000, 
            max_num_points_per_voxel=10,
            device=device
        )
        
    def generate_voxels(self, pc):
        """
        从点云数据生成体素。

        参数:
            pc (Tensor): 点云数据，形状为 (N, 3 + C)。

        返回:
            voxels (Tensor): 体素化后的特征，形状为 (num_voxels, max_points_per_voxel, C)。
            coords (Tensor): 体素坐标，形状为 (num_voxels, 3)。
            num_points_per_voxel (Tensor): 每个体素中的点数，形状为 (num_voxels)。
        """
        try:
            voxels, coords, num_points_per_voxel = self.point_to_voxel_converter(pc, empty_mean=True)
        except Exception as e:
            raise RuntimeError(f"Failed to generate voxels: {e}")
        
        return voxels, coords, num_points_per_voxel

    
class VoxelEncoder(nn.Module):
    def __init__(self, device):
        """
        初始化 VoxelEncoder。

        参数:
            device (torch.device): 运行计算的设备。
        """
        super().__init__()
        self.device = device
        self.voxelizer = Voxelizer(device)

    def forward(self, point_cloud_features):
        """
        编码批量点云特征的前向传递。

        参数:
            point_cloud_features (Tensor): 输入的点云特征，形状为 (B, N, 3 + C)。

        返回:
            all_voxels (list of Tensors): 每个批次的编码体素特征。
            all_coords (list of Tensors): 每个批次的体素坐标。
            all_num_points (list of Tensors): 每个批次中每个体素的点数。
        """
        batch_size = point_cloud_features.shape[0]
        all_voxels = []
        all_coords = []
        all_num_points = []

        # 处理批量中的每个点云
        for i in range(batch_size):
            pc = point_cloud_features[i].to(self.device)
            try:
                voxels, coords, num_points_per_voxel = self.voxelizer.generate_voxels(pc)
                encoded_features = self.encode_voxels(voxels, num_points_per_voxel)
            except Exception as e:
                raise RuntimeError(f"Failed to process batch {i}: {e}")
            
            all_voxels.append(encoded_features)
            all_coords.append(coords)
            all_num_points.append(num_points_per_voxel)
        
        return all_voxels, all_coords, all_num_points
    
    def encode_voxels(self, voxels, num_points_per_voxel):
        """


        返回:
            points_mean (Tensor): 编码后的体素特征，形状为 (num_voxels, C)。
        """
        try:
            points_mean = voxels.sum(dim=1, keepdim=False)
            normalizer = torch.clamp_min(num_points_per_voxel.view(-1, 1), min=1.0).type_as(voxels)
            points_mean = points_mean / normalizer
        except Exception as e:
            raise RuntimeError(f"\nFailed to encode voxels: {e}")
        
        return points_mean # TODO: Overload this method to include label information


#########################  Tensor相关   ######################
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
        
        try:
            spconv_tensor = spconv.SparseConvTensor(
                all_features, all_coords, spatial_shape, batch_size
            )
        except Exception as e:
            raise RuntimeError(f"\nFailed to create SparseConvTensor: {e}")
        
        return spconv_tensor

    
# PC2Tensor 类：端到端封装，输入点云特征，输出Spconv Tensor
class PC2Tensor(nn.Module):
    def __init__(self, device, spatial_shape):
        super().__init__()
        self.device = device
        self.spatial_shape = spatial_shape
        self.voxel_encoder = VoxelEncoder(device)

    def forward(self, point_cloud_features):
        try:
            encoded_features, voxel_coords, _ = self.voxel_encoder(point_cloud_features)
            spconv_tensor = TensorHelper.create_spconv_tensor(encoded_features, voxel_coords, point_cloud_features.shape[0], self.spatial_shape)
        except Exception as e:
            raise RuntimeError(f"\nFailed in PC2Tensor forward pass: {e}")
        
        return spconv_tensor


#########################  网络相关（实验性）   ######################
# SimpleSpConvNet 类， 注意不要混入真正的网络
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
        try:
            x = self.conv_input(x)
        except Exception as e:
            raise RuntimeError(f"\nFailed in SimpleSpConvNet forward pass: {e}")
        
        return x


# Example Usage
# main 函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_file = "/home/lukas/Desktop/new-script/other/try_3d/point_cloud_features_epoch_1000_batch_0.pth"
    point_cloud_features = load_pth(pth_file)
    
    print('Tensor shape', point_cloud_features.shape)
    print('Tensor type', point_cloud_features.dtype)
    
    spatial_shape = [30, 40, 40]
    pc2tensor = PC2Tensor(device, spatial_shape)
    spconv_tensor = pc2tensor(point_cloud_features)
    
    input_channels = spconv_tensor.features.shape[1]
    model = SimpleSpConvNet(input_channels, spatial_shape)
    model.to(device)
    output = model(spconv_tensor)
    
    print('Output shape:', output.features.shape)

if __name__ == "__main__":
    main()
