import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F


def project_to_3d(features, depth, K, R, T, resized_rate=1168/448, ignore_index=-1):
    """
    Project 2D features and depth map to 3D space.
    
    :param features: Tensor of shape (B, C, H, W)
    :param depth: Tensor of shape (B, H, W)
    :param K: Tensor of shape (B, 3, 3)
    :param R: Tensor of shape (B, 1, 3, 3)
    :param T: Tensor of shape (B, 1, 3)
    :param resized_rate: Scale factor to account for image resizing
    :param ignore_index: Index to ignore during projection
    :return: Tensor of shape (B, N, 3 + C) where N is the number of valid points
    """
    B, C, H, W = features.shape

    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    i = i.to(features.device)
    j = j.to(features.device)

    depth = depth.view(B, -1)  # (B, H * W)
    features = features.view(B, C, -1)  # (B, C, H * W)

    i = (resized_rate * i.view(-1).unsqueeze(0)).repeat(B, 1).float()  # (B, H * W)
    j = (resized_rate * j.view(-1).unsqueeze(0)).repeat(B, 1).float()  # (B, H * W)

    
    ones = torch.ones_like(i, dtype=torch.float).to(features.device)
    homogeneous_coords = torch.stack((j, i, ones), dim=-1)  # (B, H * W, 3)

    K_inv = torch.inverse(K)  # (B, 3, 3)
    camera_coords = torch.bmm(K_inv, homogeneous_coords.permute(0, 2, 1)) * depth.unsqueeze(1)  # (B, 3, H * W)
    
    
    # 处理R,T
    R = R.squeeze(1).transpose(1,2)  # (B, 3, 3)
    # 把R矩阵前两行乘以 -1
    R[:, 0, :] *= -1
    R[:, 1, :] *= -1
    
    # print('R:', R)  
    
    T = T.squeeze(1)  # (B, 3)
    
    # 把T的前两个元素乘以 -1
    T[:, 0] *= -1
    T[:, 1] *= -1
    
    # 好像是因为手系的问题
    

    R_inv = R.transpose(1, 2)  
    T = 1000 * (T / resized_rate)
    t_inv = -torch.bmm(R_inv, T.unsqueeze(2)).squeeze(2)  # (B, 3)
    
    world_coords = torch.bmm(R_inv, camera_coords) + t_inv.unsqueeze(2)  # (B, 3, H * W)

    # Filter out invalid points
    valid_world_coords = world_coords.permute(0, 2, 1).view(B, -1, 3)  # (B, N, 3)
    valid_features = features.permute(0, 2, 1).view(B, -1, C)  # (B, N, C)

    point_cloud = torch.cat((valid_world_coords, valid_features), dim=-1)  # (B, N, 3 + C)
    
    point_cloud[:,:,:3]/=1000.0

    return point_cloud




def apply_softmax(point_cloud_features):
    """
    Apply softmax to the feature dimensions and convert to integer predictions.
    
    :param point_cloud_features: Tensor of shape (B, N, 3 + C_classes)
    :return: Tensor of shape (B, N, 3 + 1) with integer predictions
    """
    xyz = point_cloud_features[:, :, :3]
    features = point_cloud_features[:, :, 3:]
    
    # Apply softmax to the feature dimensions
    softmaxed_features = torch.softmax(features, dim=-1)
    
    # Get the predictions
    predictions = torch.argmax(softmaxed_features, dim=-1, keepdim=True)
    
    # Combine XYZ with predictions
    point_cloud_with_predictions = torch.cat((xyz, predictions), dim=-1)
    
    return point_cloud_with_predictions

def load_palette(palette_file):
    """Load the palette file which maps labels to RGB values."""
    palette = np.loadtxt(palette_file, delimiter=' ')
    palette = palette.astype(np.uint8)
    return palette

def colorize_point_cloud(point_cloud_with_predictions, palette):
    """
    Colorize the point cloud using the predictions and the palette.
    
    :param point_cloud_with_predictions: Tensor of shape (B, N, 3 + 1)
    :param palette: Numpy array of shape (num_classes, 3)
    :return: Tensor of shape (B, N, 6) with RGB colors
    """
    xyz = point_cloud_with_predictions[:, :, :3]
    predictions = point_cloud_with_predictions[:, :, 3].long()
    
    # Convert predictions to RGB colors using the palette
    colors = torch.tensor(palette[predictions.cpu().numpy()], dtype=torch.float32).to(predictions.device) / 255.0
    
    # Combine XYZ with RGB colors
    point_cloud_with_colors = torch.cat((xyz, colors), dim=-1)
    
    return point_cloud_with_colors

def save_point_cloud_to_ply(point_cloud_with_colors, file_path):
    """
    Save the point cloud to a PLY file.
    
    :param point_cloud_with_colors: Tensor of shape (B, N, 6)
    :param file_path: Path to save the PLY file
    """
    for i in range(point_cloud_with_colors.shape[0]):
        point_cloud = point_cloud_with_colors[i].detach().cpu().numpy()
        
        # Create an open3d point cloud object
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        
        # # Save the point cloud to a PLY file
        # output_ply_path = f"{file_path}/point_cloud_{i}.ply"
        # o3d.io.write_point_cloud(output_ply_path, o3d_point_cloud)
        # print(f"Point cloud saved to {output_ply_path}")