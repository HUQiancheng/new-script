import torch
import torch.nn as nn
from transformers import SegformerModel

class Segformer_Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(Segformer_Segmentation, self).__init__()

        # Load the pre-trained Segformer model
        self.backbone = SegformerModel.from_pretrained("nvidia/mit-b1")

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.backbone.config.hidden_sizes[-1], 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(size=(224, 336), mode='bilinear', align_corners=False)  # Adjust upsampling as needed
        )

    def forward(self, x):
        # Ensure the input and model are on the same device
        device = x.device
        self.backbone.to(device)
        self.segmentation_head.to(device)
        
        #print('Input shape:', x.shape)
        # Forward pass through Segformer backbone
        outputs = self.backbone(pixel_values=x)

        # Extract the last hidden state
        x = outputs.last_hidden_state
        #print('Hidden shape:', x.shape)
        
        # Forward pass through segmentation head
        x = self.segmentation_head(x)
        #print('Output shape:', x.shape)
        return x

# Instantiate the model with 101 classes
model = Segformer_Segmentation(num_classes=101)

# Check the model architecture
print(model)
