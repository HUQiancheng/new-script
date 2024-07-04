import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path

class Model:
    def __init__(self, file_path, palette_path):
        self.data = self.load_pth_file(file_path)
        self.palette = np.loadtxt(palette_path)
        self.index = 0
    
    def load_pth_file(self, file_path):
        return torch.load(file_path)
    
    def get_image_data(self, index):
        original_image = self.data['original_image'][index]
        semantic_labels = self.data['2d_semantic_labels'][index]
        camera_params = self.data['camera_params'][index]
        
        semantic_image = self.palette[semantic_labels.clip(0)]
        semantic_image = (semantic_image * 255).astype(np.uint8)
        
        return original_image, semantic_image, camera_params
    
    def undistort_image(self, image, camera_params):
        h, w = image.shape[:2]
        K = np.array(camera_params['intrinsic_mat'])
        dist_coeffs = np.array(camera_params['radial_params']).reshape(-1)
        
        if dist_coeffs.size == 6:
            dist_coeffs = dist_coeffs[:4]  # OpenCV fisheye expects 4 coefficients

        new_K = K.copy()  # In fisheye, we usually keep the same K matrix
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, dist_coeffs, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
        
        return undistorted_image
    
    def get_undistorted_images(self, index):
        original_image, semantic_image, camera_params = self.get_image_data(index)
        undistorted_raw = self.undistort_image(original_image, camera_params)
        undistorted_sem = self.undistort_image(semantic_image, camera_params)
        return original_image, semantic_image, undistorted_raw, undistorted_sem

class View:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('Display 2D Data')
        self.next_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])
        self.prev_button_ax = self.fig.add_axes([0.7, 0.01, 0.1, 0.05])
        self.next_button = Button(self.next_button_ax, 'Next')
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.next_button.on_clicked(self.on_next_clicked)
        self.prev_button.on_clicked(self.on_prev_clicked)
        self.next_image_callback = None
        self.prev_image_callback = None
        self.loading = False
    
    def display_images(self, images, titles):
        self.loading = True
        for ax, img, title in zip(self.axes.flatten(), images, titles):
            ax.clear()
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.draw()
        self.loading = False
    
    def on_next_clicked(self, event):
        if not self.loading and self.next_image_callback:
            self.next_image_callback()
    
    def on_prev_clicked(self, event):
        if not self.loading and self.prev_image_callback:
            self.prev_image_callback()
    
    def set_next_image_callback(self, callback):
        self.next_image_callback = callback
    
    def set_prev_image_callback(self, callback):
        self.prev_image_callback = callback
    
    def show(self):
        plt.show()

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.set_next_image_callback(self.next_image)
        self.view.set_prev_image_callback(self.prev_image)
        self.current_index = 0
        self.display_current_image()
    
    def display_current_image(self):
        images = self.model.get_undistorted_images(self.current_index)
        titles = ['Distorted Raw', 'Distorted Semantic', 'Undistorted Raw', 'Undistorted Semantic']
        self.view.display_images(images, titles)
    
    def next_image(self):
        self.current_index += 1
        if self.current_index >= len(self.model.data['original_image']):
            self.current_index = 0  # Loop back to the first image
        self.display_current_image()
    
    def prev_image(self):
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = len(self.model.data['original_image']) - 1  # Loop back to the last image
        self.display_current_image()

def main(file_path, palette_path):
    model = Model(file_path, palette_path)
    view = View()
    controller = Controller(model, view)
    view.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test and display images from pth file.')
    parser.add_argument('pth_file', type=str, help='Path to the pth file.')
    parser.add_argument('palette_file', type=str, help='Path to the palette file.')
    args = parser.parse_args()
    main(args.pth_file, args.palette_file)
