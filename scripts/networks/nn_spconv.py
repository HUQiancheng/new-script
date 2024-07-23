import torch.nn as nn
import spconv.pytorch as spconv
class SimpleSpConvNet(nn.Module):
    def __init__(self, input_channels=101, num_classes=101):
        super(SimpleSpConvNet, self).__init__()
        self.encoder = spconv.SparseSequential(
            spconv.SparseConv3d(input_channels, 32, 3, 2, indice_key="conv1"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            spconv.SubMConv3d(32, 32, 3, indice_key="subm1"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.decoder = spconv.SparseSequential(
            spconv.SparseInverseConv3d(32, num_classes, 3, indice_key="conv1"), 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x