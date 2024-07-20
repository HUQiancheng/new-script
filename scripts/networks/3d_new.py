import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.utils import PointToVoxel



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
            vsize_xyz=[0.05, 0.05, 0.05],
            coors_range_xyz=[-1, -2, -2, 1, 2, 2], 
            num_point_features=101,
            max_num_voxels=2000, 
            max_num_points_per_voxel=25,
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
        voxels, coords, num_points_per_voxel = self.point_to_voxel_converter(pc, empty_mean=True)
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
            voxels, coords, num_points_per_voxel = self.voxelizer.generate_voxels(pc)
            encoded_features = self.encode_voxels(voxels, num_points_per_voxel)
            
            all_voxels.append(encoded_features)
            all_coords.append(coords)
            all_num_points.append(num_points_per_voxel)
        
        return all_voxels, all_coords, all_num_points
    
    def encode_voxels(self, voxels, num_points_per_voxel):
        """
        通过平均每个体素中的点来编码体素特征。

        参数:
            voxels (Tensor): 体素化后的特征，形状为 (num_voxels, max_points_per_voxel, C)。
            num_points_per_voxel (Tensor): 每个体素中的点数，形状为 (num_voxels)。

        返回:
            points_mean (Tensor): 编码后的体素特征，形状为 (num_voxels, C)。
        """
        points_mean = voxels.sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_points_per_voxel.view(-1, 1), min=1.0).type_as(voxels)
        points_mean = points_mean / normalizer
        return points_mean




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
        
        spconv_tensor = spconv.SparseConvTensor(
            all_features, all_coords, spatial_shape, batch_size
        )
        return spconv_tensor
    
# PC2Tensor 类：端到端封装，输入点云特征，输出Spconv Tensor
class PC2Tensor(nn.Module):
    def __init__(self, device, spatial_shape):
        super().__init__()
        self.device = device
        self.spatial_shape = spatial_shape
        self.voxel_encoder = VoxelEncoder(device)

    def forward(self, point_cloud_features):
        encoded_features, voxel_coords, _ = self.voxel_encoder(point_cloud_features)
        spconv_tensor = TensorHelper.create_spconv_tensor(encoded_features, voxel_coords, point_cloud_features.shape[0], self.spatial_shape)
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
        x = self.conv_input(x)
        return x






# Example Usage
# main 函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_file = "/home/hua/Desktop/ml3d/new-script/point_cloud_features_epoch_51_batch_0.pth"
    point_cloud_features = load_pth(pth_file)
    
    print('Tensor shape', point_cloud_features.shape)
    print('Tensor type', point_cloud_features.dtype)
    
    spatial_shape = [40, 80, 80]
    pc2tensor = PC2Tensor(device, spatial_shape)
    spconv_tensor = pc2tensor(point_cloud_features)
    
    input_channels = spconv_tensor.features.shape[1]
    model = SimpleSpConvNet(input_channels, spatial_shape)
    model.to(device)
    output = model(spconv_tensor)
    
    print('Output shape:', output.features.shape)

if __name__ == "__main__":
    main()
