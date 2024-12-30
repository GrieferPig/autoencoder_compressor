import torch
import torch.nn as nn


# Define a Basic Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


# Define a Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(ConvBlock(in_channels + i * growth_rate, growth_rate))
        self.dense_layers = nn.ModuleList(layers)

    def forward(self, x):
        inputs = [x]
        for layer in self.dense_layers:
            new_feat = layer(torch.cat(inputs, dim=1))
            inputs.append(new_feat)
        return torch.cat(inputs, dim=1)


# Define the CAFFM (Convolutional Attention Feature Fusion Module)
class CAFFM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAFFM, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c, 1, 1)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y.expand_as(x)


# Define the Full Model
class DenoisingModel(nn.Module):
    def __init__(
        self, in_channels=3, num_features=64, growth_rate=32, num_dense_layers=4
    ):
        super(DenoisingModel, self).__init__()

        # Initial Conv layer
        self.initial_conv = ConvBlock(in_channels, num_features)

        # Three Dense Blocks
        self.dense1 = DenseBlock(num_features, growth_rate, num_dense_layers)
        self.dense2 = DenseBlock(
            num_features + num_dense_layers * growth_rate, growth_rate, num_dense_layers
        )
        self.dense3 = DenseBlock(
            num_features + 2 * num_dense_layers * growth_rate,
            growth_rate,
            num_dense_layers,
        )

        # Convolutional Attention Feature Fusion Module (CAFFM)
        self.attention = CAFFM(num_features + 3 * num_dense_layers * growth_rate)

        # Final reconstruction layer
        self.reconstruction = nn.Conv2d(
            num_features + 3 * num_dense_layers * growth_rate,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # Initial Conv Layer
        out = self.initial_conv(x)

        # Dense Blocks with Residual Fusion
        out1 = self.dense1(out)
        out2 = self.dense2(out1)
        out3 = self.dense3(out2)

        # Apply Attention Mechanism
        out = self.attention(out3)

        # Final reconstruction to the same input shape
        out = self.reconstruction(out)

        return out
