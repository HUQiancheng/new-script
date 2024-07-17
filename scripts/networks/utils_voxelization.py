#############TODO: 这里需要进行点云数据的处理，将点云数据转换为体素数据，还有其他的各种相关spatial shape的处理#########
from spconv.pytorch.utils import PointToVoxel
import numpy as np
import torch
import trimesh

def load_point_cloud(ply_file):
    # Load the point cloud using trimesh
    mesh = trimesh.load(ply_file)
    # Extract the vertices (points) from the mesh
    point_cloud = mesh.vertices

    # Convert the point cloud (numpy array) to a torch tensor
    point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32)
    return point_cloud_tensor

def save_as_ply(points, filename):
    # Create a Trimesh point cloud object
    cloud = trimesh.points.PointCloud(points)

    # Export the point cloud to a PLY file
    cloud.export(filename)

def get_max_coordinate_range(point_cloud):
    # Calculate the minimum and maximum coordinates along each axis
    min_coords, _ = torch.min(point_cloud, dim=0)
    max_coords, _ = torch.max(point_cloud, dim=0)

    # Convert the results to numpy arrays for easier manipulation
    min_coords = min_coords.numpy()
    max_coords = max_coords.numpy()
    # Calculate the range along each axis
    coordinate_range = max_coords - min_coords
    return min_coords, max_coords, coordinate_range

def BatchPC2Voxel(pc, save_path):
    '''
    :param   pc: point cloud, that is ready to be voxelized [tensor](N x 3+)
             save_path: "/absolute_path/XXX.ply" for the file that you want to save the visualization for voxel [str]
    :return: voxels: [tensor](N x max_num_points_per_voxel x num_point_features)
             coords: [tensor](N x num_point_features)
             num_points_per_voxel: [tensor](N)
    A simple implementation of the PointToVoxel API from SpConv.
    Please ensure you have a version of SpConv installed on pip (This util is based on SpConv for CUDA12.0)

    When the result is incomplete, please check arguments between line 59 and 63.
    Make sure you have whole area covered && enough max_num_voxels
    '''

    # load point cloud data
    # pc = load_point_cloud(pc_path))

    min_xyz, max_xyz, range_xyz = get_max_coordinate_range(pc)

    print(f"Minimum coordinates: {min_xyz}")
    print(f"Maximum coordinates: {max_xyz}")
    print(f"Coordinate range: {range_xyz}")
    coors_range_xyz = []
    safe_min = min_xyz - range_xyz
    safe_max = max_xyz + range_xyz
    for min in safe_min:
        coors_range_xyz.append(min)
    for max in safe_max:
        coors_range_xyz.append(max)
    max_num_voxels = int(np.prod(range_xyz)*8000)
    gen = PointToVoxel(
        vsize_xyz=[0.05, 0.05, 0.05],
        coors_range_xyz=coors_range_xyz,
        num_point_features=3,  # ? Related to feature channels
        max_num_voxels=max_num_voxels,  # CHANGE here, when the result doesn't include the whole scene
        max_num_points_per_voxel=10)
    voxels, coords, num_points_per_voxel = gen(pc, empty_mean=True)

    # print voxel summary:
    print("number of voxels", coords.shape[0])
    num_points = num_points_per_voxel.numpy()
    ar, num = np.unique(num_points, return_counts=True)
    i = 0
    for element in ar:
        print("number of voxels with", element, "points:", num[i])
        i += 1

    # save point cloud data
    save_as_ply(coords, save_path)

    return voxels, coords, num_points_per_voxel

pc_path = '/home/john/repos/pc_aligned.ply'
save_path = '/home/john/repos/pc_aligned_voxel.ply'
pointcloud = load_point_cloud(pc_path)
BatchPC2Voxel(pointcloud, save_path)


'''
# If you want to get label for every point of your pc, you need to use another function to get pc_voxel_id and gather features from sematic segmentation result:

voxels, coords, num_points_per_voxel, pc_voxel_id = gen.generate_voxel_with_id(pc_th, empty_mean=True)
seg_features = YourSegNet(...)
# if voxel id is invalid (point out of range, or no space left in a voxel),
# features will be zero.
point_features = gather_features_by_pc_voxel_id(seg_features, pc_voxel_id)
'''