import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import open3d as o3d
import numpy as np

class SimpleSpConvNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SimpleSpConvNet, self).__init__()
        self.encoder = spconv.SparseSequential(
            spconv.SparseConv3d(input_channels, 64, 3, 2, indice_key="conv1"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, indice_key="subm1"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.decoder = spconv.SparseSequential(
            spconv.SparseInverseConv3d(64, output_channels, 3, indice_key="conv1"),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_spconv_tensor(pth_file, device):
    input = torch.load(pth_file, map_location=device)
    return input

def preprocess_labels(input_tensor):
    features = input_tensor.features
    softmax_features = F.softmax(features, dim=1)
    labels = torch.argmax(softmax_features, dim=1)
    return labels

def create_voxel_grid(indices, labels, voxel_size=0.05):
    max_label = labels.max().item()
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = voxel_size
    
    for i, label in enumerate(labels):
        voxel = o3d.geometry.Voxel(grid_index=indices[i, 1:4])
        voxel.color = [label.item() / max_label, 0, 1 - label.item() / max_label]
        voxel_grid.voxels.append(voxel)
    
    return voxel_grid

def visualize_labels_as_voxels(indices, labels, voxel_size=0.05):
    max_label = labels.max().item()
    colors = np.zeros((labels.shape[0], 3))
    for i in range(labels.shape[0]):
        colors[i] = [labels[i] / max_label, 0, 1 - labels[i] / max_label]
    
    points = indices[:, 1:4] * voxel_size  # Skip batch index and apply voxel size

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pth_file = "/home/lukas/Desktop/new-script/other/try_3d_backbone/features/spconv_tensor_epoch_500_batch_0.pth"
    input = load_spconv_tensor(pth_file, device)
    
    labels = preprocess_labels(input)
    
    print('Spatial shape:', input.spatial_shape)
    print('Indices shape:', input.indices.shape)
    print('Features shape:', input.features.shape)
    
    model = SimpleSpConvNet(input_channels=input.features.shape[1], output_channels=labels.max().item() + 1).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Ensure input and labels are SparseConvTensor
    labels_sparse_tensor = spconv.SparseConvTensor(
        labels.unsqueeze(1).float(), input.indices, input.spatial_shape, input.batch_size
    )
    
    model.train()
    for epoch in range(10):  # Number of epochs for debugging
        optimizer.zero_grad()
        output = model(input)
        output_dense = output.dense()
        
        # Flatten the tensors for loss computation
        output_flat = output_dense.view(-1, output_dense.shape[1])
        labels_flat = labels_sparse_tensor.dense().view(-1).long()
        
        loss = criterion(output_flat, labels_flat)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
    
    print('Output shape:', output.dense().shape)
    
    # Visualize label voxels
    voxel_grid = visualize_labels_as_voxels(input.indices.cpu().numpy(), labels.cpu().numpy())
    o3d.visualization.draw_geometries([voxel_grid])

if __name__ == "__main__":
    main()
