import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
import open3d as o3d
import numpy as np
from utils_voxelization import PC2Tensor, VoxelEncoder,load_pth
from print_summary import summarize_pth_file

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
    

        
def main():
    pth_path = '/home/lukas/Desktop/new-script/other/try_3d_backbone/data/0a5c013435.pth'
    data = load_pth(pth_path)
    
    # created needed pc2tensor input
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # summarize_pth_file(pth_path) # to save a summary of the .pth file
    # get coords and labels
    raw_coords = data['vtx_coords']
    raw_colors = data['vtx_colors']
    raw_labels = data['vtx_labels']
    
    print('Raw coords shape', raw_coords.shape)
    print('Raw colors shape', raw_colors.shape)
    print('Raw labels shape', raw_labels.shape)
    # Convert to tensor
    pc_colors = torch.tensor(raw_colors, dtype=torch.float32).to(device)
    pc_labels = torch.tensor(raw_labels, dtype=torch.long).to(device).
    pc_coords = torch.tensor(raw_coords, dtype=torch.float32).to(device)
    # Add batch dimension
    pc_colors = pc_colors.unsqueeze(0)
    pc_labels = pc_labels.unsqueeze(0)
    pc_coords = pc_coords.unsqueeze(0)
    # Concatenate coords and colors along the last axis
    pc_colors = torch.cat((pc_coords, pc_colors), dim=2)
    pc_labels = torch.cat((pc_coords, pc_labels), dim=2)
    
    
    
    
    
    print('Colored Tensor shape', pc_colors.shape)
    print('Labels Tensor shape', pc_labels.shape)
    

if __name__ == "__main__":
    main()
    

