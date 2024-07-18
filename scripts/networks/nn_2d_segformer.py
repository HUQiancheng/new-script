import torch
import torch.nn as nn
from transformers import SegformerModel

# 如果要更换模型需要修改的地方：
# 1. "nvidia/mit-b1" 改为 "nvidia/mit-b0"
# 2. 也是backbone需要分别除以2，见被注释掉的部分
# 3. 最重要的是，如果你需要更换模型，请确保你的checkpoints已经清空，否则会加载之前的，模型结构不匹配

class Segformer_Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(Segformer_Segmentation, self).__init__()

        # # Load the pre-trained Segformer model
        # self.backbone = SegformerModel.from_pretrained("nvidia/mit-b1", output_hidden_states=True) 
        
        # Load this pre-trained Segformer model if not enough Memory
        self.backbone = SegformerModel.from_pretrained("nvidia/mit-b0", output_hidden_states=True) 

        # # Segmentation head using MLP layers
        # self.linear_c4 = nn.Linear(512, 128)  # assuming backbone.config.hidden_sizes[-1] is 128
        # self.linear_c3 = nn.Linear(320, 128)
        # self.linear_c2 = nn.Linear(128, 128)
        # self.linear_c1 = nn.Linear(64, 128)
        
        # Modify the channel dimensions as if you changed to "nvidia/mit-b0" like this:
        self.linear_c4 = nn.Linear(256, 128)  # assuming backbone.config.hidden_sizes[-1] is 128
        self.linear_c3 = nn.Linear(160, 128)
        self.linear_c2 = nn.Linear(64, 128)
        self.linear_c1 = nn.Linear(32, 128)

        self.relu = nn.ReLU(inplace=True)
        self.upsample1 = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample4 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        

        self.concat_linear = nn.Linear(4 * 128, 128)  # 4 * 128 channels after concatenation
        self.upsample_cat = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.output_linear = nn.Linear(128, num_classes)

    def forward(self, x):

        # Forward pass through Segformer backbone
        outputs = self.backbone(pixel_values=x)

        # Extract multi-level features
        c1, c2, c3, c4 = outputs.hidden_states
        
        # print('Shape of four hidden layers: (N, C, H, W)\n', c1.shape, c2.shape, c3.shape, c4.shape)

        # Apply MLP to unify channel dimensions
        c1 = self.linear_c1(c1.permute(0, 2, 3, 1))  # (N, H, W, C)
        c1 = self.relu(c1)
        c2 = self.linear_c2(c2.permute(0, 2, 3, 1))  # (N, H, W, C)
        c2 = self.relu(c2)
        c3 = self.linear_c3(c3.permute(0, 2, 3, 1))  # (N, H, W, C)
        c3 = self.relu(c3)
        c4 = self.linear_c4(c4.permute(0, 2, 3, 1))  # (N, H, W, C)
        c4 = self.relu(c4)
        
        # print('Shape of four hidden layers after MLP: (N, H, W, C)\n', c1.shape, c2.shape, c3.shape, c4.shape)
        # Upsample features
        c1 = self.upsample1(c1.permute(0, 3, 1, 2))  # (N, C, H, W)
        c2 = self.upsample2(c2.permute(0, 3, 1, 2))  # (N, C, H, W)
        c3 = self.upsample3(c3.permute(0, 3, 1, 2))  # (N, C, H, W)
        c4 = self.upsample4(c4.permute(0, 3, 1, 2))  # (N, C, H, W)
        
        # print('Shape of four hidden layers after upsampling: (N, C, H, W)\n', c1.shape, c2.shape, c3.shape, c4.shape)
        # Concatenate features
        fused = torch.cat([c1, c2, c3, c4], dim=1)  # (N, 4*C, H, W)
        
        # print('Shape of concatenated features: (N, 4*C, H, W)\n', fused.shape)
        # Apply final MLP layers
        fused = fused.permute(0, 2, 3, 1)  # (N, H, W, 4*C)
        fused = self.concat_linear(fused)
        fused = self.relu(fused)
        fused = self.upsample_cat(fused.permute(0, 3, 1, 2))  # (N, C, H, W)
        fused = self.output_linear(fused.permute(0, 2, 3, 1))  # (N, H, W, num_classes)

        fused = fused.permute(0, 3, 1, 2)  # (N, num_classes, H, W)

        return fused

