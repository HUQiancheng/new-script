import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def load_palette(palette_file):
    with open(palette_file, 'r') as f:
        palette = f.read().splitlines()
    palette = [list(map(lambda x: int(float(x)), color.split())) for color in palette]
    palette = np.array(palette) / 255.0  # Normalize palette to [0, 1]
    print(f"Loaded palette: {palette[:5]}")  # Debugging print
    return palette

def visualize_predictions(inputs, labels, predictions, palette_file, num_samples=4):
    palette = load_palette(palette_file)
    indices = random.sample(range(len(inputs)), min(num_samples, len(inputs)))
    
    fig, axs = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    if len(indices) == 1:
        axs = [axs]
    
    for i, idx in enumerate(indices):
        input_img = inputs[idx][0].permute(1, 2, 0).cpu().numpy()  # [0] to get the image from the batch
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min()) # Normalize to [0, 1]
        label_img = labels[idx][0].cpu().numpy()
        pred_img = predictions[idx][0].argmax(dim=0).cpu().numpy()
        
        print(f"Input image shape: {input_img.shape}, min: {input_img.min()}, max: {input_img.max()}")
        print(f"Label image shape: {label_img.shape}, unique values: {np.unique(label_img)}")
        print(f"Prediction shape: {pred_img.shape}, unique values: {np.unique(pred_img)}")
        
        label_img_color = np.zeros((*label_img.shape, 3), dtype=np.float32)
        pred_img_color = np.zeros((*pred_img.shape, 3), dtype=np.float32)
        
        for idx, color in enumerate(palette):
            if idx in np.unique(label_img):  # Only map colors for present labels
                label_img_color[label_img == idx] = color
            if idx in np.unique(pred_img):  # Only map colors for present predictions
                pred_img_color[pred_img == idx] = color
        
        axs[i][0].imshow(input_img)
        axs[i][0].set_title('Input Image')
        axs[i][1].imshow(label_img_color)
        axs[i][1].set_title('Ground Truth')
        axs[i][2].imshow(pred_img_color)
        axs[i][2].set_title('Prediction')    
    plt.show()
    
def visualize_dataloader(dataloader, palette_file, num_samples=4):
    palette = load_palette(palette_file)
    num_images = 0
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axs = [axs]
    
    for i, images in enumerate(dataloader):
        if num_images >= num_samples:
            break
        inputs = images['image']
        labels = images['label']
        
        input_img = inputs[0].permute(1, 2, 0).cpu().numpy()  # Ensure correct normalization if needed
        
        # Normalize the input image to [0, 1]
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
        
        label_img = labels[0].cpu().numpy()
        
        print(f"Input image shape: {input_img.shape}, min: {input_img.min()}, max: {input_img.max()}")
        print(f"Label image shape: {label_img.shape}, unique values: {np.unique(label_img)}")
        
        label_img_color = np.zeros((*label_img.shape, 3), dtype=np.float32)
        
        for idx, color in enumerate(palette):
            if idx in np.unique(label_img):  # Only map colors for present labels
                label_img_color[label_img == idx] = color
        
        axs[num_images][0].imshow(input_img)
        axs[num_images][0].set_title('Input Image')
        axs[num_images][1].imshow(label_img_color)
        axs[num_images][1].set_title('Ground Truth')
        
        num_images += 1
    
    # Remove extra subplots if any
    for j in range(num_images, num_samples):
        fig.delaxes(axs[j][0])
        fig.delaxes(axs[j][1])
    
    plt.show()
