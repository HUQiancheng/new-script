import torch
import numpy as np

def summarize_pth_file(file_path):
    # Load the .pth file
    data = torch.load(file_path)
    
    # Initialize the summary
    summary = []

    # Extract and summarize properties
    summary.append(f"Summary of .pth file: {file_path}\n")
    summary.append("Keys and corresponding data shapes and types:\n")
    
    for key, value in data.items():
        if isinstance(value, list) and len(value) > 0:
            summary.append(f"{key}: list of {type(value[0])}, length = {len(value)}")
            if isinstance(value[0], torch.Tensor):
                summary.append(f"    Shape of first element: {value[0].shape}")
                summary.append(f"    Data type of first element: {value[0].dtype}")
            elif isinstance(value[0], np.ndarray):
                summary.append(f"    Shape of first element: {value[0].shape}")
                summary.append(f"    Data type of first element: {value[0].dtype}")
            elif isinstance(value[0], dict) and key == 'camera_params':
                # Summarize the structure of camera_params
                summary.append(f"    Structure of first element:")
                for cam_key, cam_value in value[0].items():
                    if isinstance(cam_value, np.ndarray):
                        summary.append(f"        {cam_key}: numpy.ndarray, shape = {cam_value.shape}, dtype = {cam_value.dtype}")
                    else:
                        summary.append(f"        {cam_key}: {type(cam_value)}")
        elif isinstance(value, dict):
            summary.append(f"{key}: dictionary with {len(value)} keys")
        else:
            summary.append(f"{key}: {type(value)}")

    # Convert summary to string
    summary_str = "\n".join(summary)
    
    # Save under "data_descriptions" folder directly(not with join exisiting path)
    summary_file_path = "other/pth_summary.txt"
    
    # If path does not exist, create it
    import os
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    # Write the summary to the file
    with open(summary_file_path, "w") as file:
        file.write(summary_str)

    print(f"Summary saved to {summary_file_path}")

# Example usage
file_path = "dataset/data/a24f64f7fb.pth"
summarize_pth_file(file_path)
