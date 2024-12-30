import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        image_size=128,
        num_layers=3,
        latent_dim=256,
    ):
        """
        Initializes the Autoencoder.

        Parameters:
        - input_channels (int): Number of channels in the input images.
        - output_channels (int): Number of channels in the output images.
        - image_size (int): Height and width of the input images (assumes square images).
        - num_layers (int): Number of convolutional layers in the encoder and decoder.
        - latent_dim (int): Dimension of the latent space.
        """
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Calculate the size after downsampling
        assert (
            image_size % (2**num_layers) == 0
        ), "Image size must be divisible by 2^num_layers"
        final_size = image_size // (2**num_layers)

        # Encoder
        encoder_layers = []
        channels = 32  # Starting number of channels
        current_channels = input_channels
        for _ in range(num_layers):
            encoder_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            encoder_layers.append(nn.ReLU(inplace=True))
            current_channels = channels
            channels = channels * 2  # Double the number of channels each layer

        encoder_layers.append(nn.Flatten())
        encoder_layers.append(
            nn.Linear(current_channels * final_size * final_size, latent_dim)
        )
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
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
                nn.ConvTranspose2d(
                    in_channels=current_channels,
                    out_channels=channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            decoder_layers.append(nn.ReLU(inplace=True))
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
        """
        Encodes the input into the latent space.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
        - latent (torch.Tensor): Latent representation of shape (batch_size, latent_dim).
        """
        latent = self.encoder(x)
        return latent

    def decode(self, latent):
        """
        Decodes the latent representation back to the original input space.

        Parameters:
        - latent (torch.Tensor): Latent representation of shape (batch_size, latent_dim).

        Returns:
        - reconstructed (torch.Tensor): Reconstructed tensor of shape (batch_size, output_channels, height, width).
        """
        reconstructed = self.decoder(latent)
        return reconstructed

    def forward(self, x):
        """
        Forward pass of the Autoencoder.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
        - reconstructed (torch.Tensor): Reconstructed tensor of shape (batch_size, output_channels, height, width).
        - latent (torch.Tensor): Latent representation of shape (batch_size, latent_dim).
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
