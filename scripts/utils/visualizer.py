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
    
    fig, axs = plt.subplots(2*len(indices), 3, figsize=(15, 7 * len(indices)))
    
    
    for i, idx in enumerate(indices):
        input_img1 = inputs[idx][0][0].permute(1, 2, 0).cpu().numpy()  # [0] to get the image from the batch
        input_img2 = inputs[idx][0][1].permute(1, 2, 0).cpu().numpy()
        input_img1 = (input_img1 - input_img1.min()) / (input_img1.max() - input_img1.min()) # Normalize to [0, 1]
        input_img2 = (input_img2 - input_img2.min()) / (input_img2.max() - input_img2.min()) # Normalize to [0, 1]
        label_img1 = labels[idx][0][0].cpu().numpy()
        label_img2 = labels[idx][0][1].cpu().numpy()
        pred_img1 = predictions[idx][0][0].argmax(dim=0).cpu().numpy()
        pred_img2 = predictions[idx][0][1].argmax(dim=0).cpu().numpy()
        
        print(f"Input image shape: {input_img1.shape}, min: {input_img1.min()}, max: {input_img1.max()}")
        print(f"Label image shape: {label_img1.shape}, unique values: {np.unique(label_img1)}")
        print(f"Prediction image shape: {pred_img1.shape}, unique values: {np.unique(pred_img1)}")
        
        label_img_color1 = np.zeros((*label_img1.shape, 3), dtype=np.float32)
        pred_img_color1 = np.zeros((*pred_img1.shape, 3), dtype=np.float32)
        label_img_color2 = np.zeros((*label_img2.shape, 3), dtype=np.float32)
        pred_img_color2 = np.zeros((*pred_img2.shape, 3), dtype=np.float32)

        for idx, color in enumerate(palette):
            if idx in np.unique(label_img1):  # Only map colors for present labels
                label_img_color1[label_img1 == idx] = color
            if idx in np.unique(pred_img1):  # Only map colors for present predictions
                pred_img_color1[pred_img1 == idx] = color
        for idx, color in enumerate(palette):
            if idx in np.unique(label_img2):  # Only map colors for present labels
                label_img_color2[label_img2 == idx] = color
            if idx in np.unique(pred_img1):  # Only map colors for present predictions
                pred_img_color2[pred_img2 == idx] = color

        axs[i][0].imshow(input_img1)
        axs[i][0].set_title('Input Image1')
        axs[i][1].imshow(label_img_color1)
        axs[i][1].set_title('Ground Truth1')
        axs[i][2].imshow(pred_img_color1)
        axs[i][2].set_title('Prediction1')  

        axs[i+1][0].imshow(input_img2)
        axs[i+1][0].set_title('Input Image2')
        axs[i+1][1].imshow(label_img_color2)
        axs[i+1][1].set_title('Ground Truth2')
        axs[i+1][2].imshow(pred_img_color2)
        axs[i+1][2].set_title('Prediction2')  
    
    plt.show()
    
def visualize_dataloader(dataloader, palette_file, num_samples=4):
    palette = load_palette(palette_file)
    num_images = 0
    fig, axs = plt.subplots(num_samples, 4, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axs = [axs]
    
    for i, images in enumerate(dataloader):
        if num_images >= num_samples:
            break
        inputs = images['image']
        labels = images['label']
        
        # Assuming inputs is of shape (batch_size, 2, C, H, W)
        img1 = inputs[0][0].permute(1, 2, 0).cpu().numpy()
        img2 = inputs[0][1].permute(1, 2, 0).cpu().numpy()
        
        # Normalize the input images to [0, 1]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        
        label_img1 = labels[0][0].cpu().numpy()
        label_img2 = labels[0][1].cpu().numpy()
        
        print(f"Input image 1 shape: {img1.shape}, min: {img1.min()}, max: {img1.max()}")
        print(f"Input image 2 shape: {img2.shape}, min: {img2.min()}, max: {img2.max()}")
        print(f"Label image shape: {label_img1.shape}, unique values: {np.unique(label_img1)}")
        print(f"Label image shape: {label_img2.shape}, unique values: {np.unique(label_img2)}")
        
        label_img1_color = np.zeros((*label_img1.shape, 3), dtype=np.float32)
        label_img2_color = np.zeros((*label_img2.shape, 3), dtype=np.float32)
        
        for idx, color in enumerate(palette):
            if idx in np.unique(label_img1):  # Only map colors for present labels
                label_img1_color[label_img1 == idx] = color
            if idx in np.unique(label_img2):  # Only map colors for present labels
                label_img2_color[label_img2 == idx] = color
        
        axs[num_images][0].imshow(img1)
        axs[num_images][0].set_title('Input Image 1')
        axs[num_images][1].imshow(label_img1_color)
        axs[num_images][1].set_title('Ground Truth 1')
        
        axs[num_images][2].imshow(img2)
        axs[num_images][2].set_title('Input Image 2')
        axs[num_images][3].imshow(label_img2_color)
        axs[num_images][3].set_title('Ground Truth 2')
        
        num_images += 1
    
    # Remove extra subplots if any
    for j in range(num_images, num_samples):
        for k in range(4):
            fig.delaxes(axs[j][k])
    
    plt.show()
    