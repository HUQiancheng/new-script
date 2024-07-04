# 2D and 3D Data Processing and Visualization

## Overview

This 2D dataset involves processing 3D semantics onto DSLR images by rasterizing the mesh and then visualizing the results. The data is saved in `.pth` files with specific structures, and a testing script is provided to visualize the processed data.

## Data Structure

Each `.pth` file contains a dictionary with the following keys:
- `image_name`: List of all image names as strings.
- `original_image`: List of all original images.
- `2d_semantic_labels`: List of all semantic label images.
- `camera_params`: List of dictionaries containing the camera parameters for each corresponding image:
  - `focal_length`: Focal length of the camera.
  - `principal_point`: Principal point of the camera.
  - `radial_params`: Radial distortion parameters.
  - `R`: Rotation matrix.
  - `T`: Translation vector.
  - `use_radial`: Boolean indicating if radial distortion is used.
  - `intrinsic_mat`: Intrinsic matrix of the camera.
  - `extrinsic`: Extrinsic matrix (camera pose).
  - `view_width_px`: Width of the image in pixels.
  - `view_height_px`: Height of the image in pixels.

## Getting Started

### Rasterize 3D Semantics onto 2D Images
To get 3D semantics onto 2D DSLR images, run 
```
python -m semantic.prep.rasterize_semantics_2d semantic/configs/rasterize_semantics_2d.yml
```

Modify that script to save anything necessary after ```pix_sem_ids``` is obtained.

### Testing and Visualization

To test and visualize the processed data, run the following command:
```bash
python test_pth.py path_to_your_pth_file.pth path_to_your_palette_file.txt
```
To get description, run the following command:

```bash
python utils/print_summary.py
```

### Example for visualization

```bash
python utils/test_pth.py dataset/data8b5caf3398.pth utils/palette_scannet200.txt
```

### Required Libraries

Ensure you have the following libraries installed:

- torch
- numpy
- cv2 (OpenCV)
- matplotlib

You can install the necessary libraries using pip, for example:

```bash
pip install torch numpy opencv-python matplotlib 
```
