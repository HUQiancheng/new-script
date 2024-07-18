import torch.nn as nn
import spconv.pytorch as spconv
class SimpleSpConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_input = spconv.SparseSequential(
            spconv.SparseConv3d(num_classes, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 32, 3, padding=1),
        )
        
    def forward(self, x):
        try:
            x = self.conv_input(x)
        except Exception as e:
            raise RuntimeError(f"\nFailed in SimpleSpConvNet forward pass: {e}")
        
        return x