import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, transpose=False):
        super(ResidualBlock, self).__init__()
        self.transpose = transpose
        self.stride = stride

        # Main convolutional layer
        self.conv = (
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                output_padding=0,
            )
            if transpose
            else nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=stride, padding=1
            )
        )
        self.activation = nn.ReLU(inplace=True)

        # Adjust residual connection to match spatial dimensions and channels
        if transpose:
            if in_channels != out_channels or stride != 1:
                # Calculate appropriate output_padding
                # Typically, output_padding=1 when stride=2 to align dimensions
                output_padding = stride - 1 if stride > 1 else 0
                self.residual = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    output_padding=output_padding,
                )
            else:
                self.residual = nn.Identity()
        else:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, padding=0
                )
            else:
                self.residual = nn.Identity()

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = self.residual(x)  # Adjust residual if needed
        out = self.conv(x)
        out = self.activation(out)

        return self.skip_add.add(out, identity)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        image_size=128,
        num_layers=3,
        latent_dim=256,
    ):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Calculate the size after downsampling
        assert image_size % (2**num_layers) == 0, "non integer downsampling"
        final_size = image_size // (2**num_layers)

        # Encoder with residual connections
        encoder_layers = []
        channels = 32  # Starting number of channels
        current_channels = input_channels
        for _ in range(num_layers):
            encoder_layers.append(
                ResidualBlock(
                    in_channels=current_channels,
                    out_channels=channels,
                    stride=2,
                )
            )
            current_channels = channels
            channels = channels * 2  # Double the number of channels each layer

        encoder_layers.append(nn.Flatten())
        encoder_layers.append(
            nn.Linear(current_channels * final_size * final_size, latent_dim)
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder with residual connections
        decoder_layers = []
        decoder_layers.append(
            nn.Linear(latent_dim, current_channels * final_size * final_size)
        )
        decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(
            nn.Unflatten(
                dim=1, unflattened_size=(current_channels, final_size, final_size)
            )
        )

        for _ in range(num_layers - 1):
            channels = current_channels // 2  # Halve the number of channels each layer
            decoder_layers.append(
                ResidualBlock(
                    in_channels=current_channels,
                    out_channels=channels,
                    stride=2,
                    transpose=True,
                )
            )
            current_channels = channels

        # Final layer to get back to output channels
        decoder_layers.append(
            nn.ConvTranspose2d(
                in_channels=current_channels,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        decoder_layers.append(nn.Sigmoid())  # Ensures output is between 0 and 1

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):

        latent = self.encoder(x)
        return latent

    def decode(self, latent):

        reconstructed = self.decoder(latent)
        return reconstructed

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
