import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3_Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3_Segmentation, self).__init__()
        self.backbone = models.mobilenet_v3_small(pretrained=True).features
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(576, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout with a probability of 0.3
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Dropout with a probability of 0.2
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.segmentation_head(x)
        return x
