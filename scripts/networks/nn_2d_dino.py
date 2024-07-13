import torch
import torch.nn as nn
import torch.hub
# Also clear the shitty warnings
import warnings

# Suppress specific warning
warnings.filterwarnings(
    "ignore", 
    message="The default value of the antialias parameter of all the resizing transforms"
)

import torch
import torch.nn as nn
import torch.hub

class DINOv2_Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(DINOv2_Segmentation, self).__init__()
        
        # Load the pre-trained DINOv2 model
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        
        # Segmentation head with multiple upsampling steps
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(2, 3), mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=7, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False),
        
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Forward pass through ViT
        x = self.backbone(x)
        
        # Reshape ViT output to feature map (batch_size, 1, 16, 24)
        x = x.view(batch_size, 384, 1, 1)
        
        # Forward pass through segmentation head
        x = self.segmentation_head(x)
        return x

