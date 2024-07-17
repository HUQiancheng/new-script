import torch
import numpy as np
import open3d as o3d

class DataLoader:
    @staticmethod
    def load_pth_file(file_path):
        """Load the .pth file."""
        return torch.load(file_path)

class Camera:
    def __init__(self, camera_params):
        self.K = camera_params['intrinsic_mat']
        extrinsic = camera_params['extrinsic']
        self.R = extrinsic[:3, :3]
        self.t = extrinsic[:3, 3]
    
    def get_intrinsics(self):
        return self.K
    
    def get_extrinsics(self):
        return self.R, self.t

class PointCloud3D:
    def __init__(self, depth_image, label_image, camera, resized_rate=1168/448):
        self.depth_image = depth_image
        self.label_image = label_image
        self.camera = camera
        self.resized_rate = resized_rate
    
    def generate_point_cloud(self):
        H, W = self.depth_image.shape
        i, j = np.indices((H, W))

        # Flatten the indices and depth values, and consider the resized rate
        i = self.resized_rate * i.flatten().astype(np.int32)
        j = self.resized_rate * j.flatten().astype(np.int32)
        depth = self.depth_image.flatten()

        # Create homogeneous coordinates
        ones = np.ones_like(depth)
        homogeneous_coords = np.stack((j, i, ones), axis=-1)
        #

        # Convert pixel coordinates to camera coordinates
        K = self.camera.get_intrinsics()
        camera_coords = np.linalg.inv(K) @ homogeneous_coords.T * depth
        camera_coords = camera_coords.T

        # Convert camera coordinates to world coordinates
        R, t = self.camera.get_extrinsics()
        R_inv = R.T
        t = 1000 * (t / self.resized_rate)
        t_inv = -R_inv @ t
        world_coords = (R_inv @ camera_coords.T + t_inv.reshape(-1, 1)).T

        labels = self.label_image.reshape(-1, 1)
        point_cloud = np.hstack((world_coords, labels))
        
        print('shape of every variables in PointCloud3D')
        print(f'i: {i.shape}, j: {j.shape}, depth: {depth.shape}, homogeneous_coords: {homogeneous_coords.shape}, camera_coords: {camera_coords.shape}, world_coords: {world_coords.shape}, labels: {labels.shape}')
        
        return point_cloud

class ColorCloud3D:
    def __init__(self, palette_file_path):
        self.palette = self.load_palette(palette_file_path)
    
    @staticmethod
    def load_palette(file_path):
        palette = np.loadtxt(file_path, delimiter=' ')
        return palette.astype(np.uint8)
    
    def transform_point_cloud(self, point_cloud_features):
        coords = point_cloud_features[:, :3]
        labels = point_cloud_features[:, 3].astype(np.int16)
        colors = self.palette[labels] / 255.0
        transformed_features = np.hstack((coords, colors))
        return transformed_features
    
    def save_point_cloud(self, point_cloud, output_path):
        o3d_point_cloud = o3d.geometry.PointCloud()
        o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        o3d.io.write_point_cloud(output_path, o3d_point_cloud)
        print(f"Point cloud saved to {output_path}")
    
class ViewColorPC:
    @staticmethod
    def visualize(point_cloud):
        o3d.visualization.draw_geometries([point_cloud])

def main(file_path, output_dir, palette_file_path):
    data = DataLoader.load_pth_file(file_path)
    all_points = []
    all_feature_colors = []

    for i in range(len(data['image_name'])):
        depth_image = data['depth_image'][i]
        label_image = data['2d_semantic_labels'][i].astype(np.int16)
        camera_params = data['camera_params'][i]
        
        camera = Camera(camera_params)
        point_cloud3d = PointCloud3D(depth_image, label_image, camera)
        point_cloud_with_labels = point_cloud3d.generate_point_cloud()

        color_cloud3d = ColorCloud3D(palette_file_path)
        point_cloud = color_cloud3d.transform_point_cloud(point_cloud_with_labels)
        
        
        point_cloud[:, :3] /= 1000.0
        
        output_ply_path = f"{output_dir}/point_cloud_{i}.ply"
        color_cloud3d.save_point_cloud(point_cloud, output_ply_path)
        
        all_points.append(point_cloud[:, :3])
        all_feature_colors.append(point_cloud[:, 3:])

    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_feature_colors)
    
    combined_point_cloud = np.hstack((combined_points, combined_colors))
    combined_output_ply_path = f"{output_dir}/combined_point_cloud.ply"
    color_cloud3d.save_point_cloud(combined_point_cloud, combined_output_ply_path)

# Example usage
if __name__ == "__main__":
    file_path = "/home/lukas/Desktop/new-script/try_proj/8b5caf3398.pth"
    output_dir = "/home/lukas/Desktop/new-script/try_proj"
    palette_file_path = "/home/lukas/Desktop/new-script/try_proj/palette_scannet200.txt"
    main(file_path, output_dir, palette_file_path)
