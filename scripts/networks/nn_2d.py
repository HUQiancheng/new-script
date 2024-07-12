import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3_Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3_Segmentation, self).__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True).features
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),  # MobileNetV3 large has 960 output channels
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),  # Output num_classes channels
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)  # Upsample to the original image size
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.segmentation_head(x)
        return x

# Instantiate the model with 100 classes
model = MobileNetV3_Segmentation(num_classes=100)
