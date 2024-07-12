import torch
import torch.nn as nn
from functools import partial
from dinov2.eval.linear import create_linear_input
from dinov2.eval.utils import ModelWithIntermediateLayers

class DINOv2_Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(DINOv2_Segmentation, self).__init__()
        
        # Load the DINOv2 model
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc', pretrained=True)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
        self.backbone = ModelWithIntermediateLayers(model, n_last_blocks=1, autocast_ctx=autocast_ctx)
        
        # Define the segmentation head
        sample_output = self.backbone(torch.randn(1, 3, 224, 224))
        out_dim = create_linear_input(sample_output, use_n_blocks=1, use_avgpool=True).shape[1]
        
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(out_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        x = create_linear_input(features, use_n_blocks=1, use_avgpool=True)
        x = self.segmentation_head(x)
        return x


