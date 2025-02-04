import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A convolutional block with optional batch normalization and activation functions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        discriminator (bool): If True, applies LeakyReLU instead of PReLU (used in the discriminator).
        apply_activation (bool): Applies an activation function if True.
        apply_normalization (bool): Applies batch normalization if True.
    """
    def __init__(self, in_channels, out_channels, discriminator=False, apply_activation=True, apply_normalization=True, **kwargs):
        super().__init__()
        self.apply_activation = apply_activation
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not apply_normalization)
        self.bn = nn.BatchNorm2d(out_channels) if apply_normalization else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.apply_activation else self.bn(self.cnn(x))

    
class UpsampleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle to increase resolution.
    
    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Factor by which the spatial resolution is increased.
    """
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)  # in_channels * 4, H, W --> in_channels, H*2, W*2
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual block used in the generator to enhance feature extraction.
    
    Helps stabilize training by allowing gradients to flow through shortcut connections.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, apply_activation=False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    """
    Generator model for SRGAN to upscale low-resolution images to high resolution.
    
    Consists of an initial convolution, multiple residual blocks, feature refinement, upsampling, and final output processing.
    
    Args:
        in_channels (int): Number of input channels (usually 3 for RGB).
        num_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks.
    """
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        
        # Step 1: Initial convolution to process the input.
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, apply_normalization=False)

        # Step 2: A stack of residual blocks to extract features.
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])

        # Step 3: Final convolution to refine the features.
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, apply_activation=False)

        # Step 4: Upsampling to increase the resolution.
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))

        # Step 5: A convolution to output the final image, followed by a tanh to scale the output between -1 and 1.
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial
        x = self.upsamples(x)
        return torch.tanh(self.final(x))


class Discriminator(nn.Module):
    """
    Discriminator model for SRGAN to classify images as real or fake.
    
    Extracts features using convolutional layers and reduces spatial dimensions before fully connected classification.
    
    Args:
        in_channels (int): Number of input channels.
        features (list): List of feature channels for each convolutional block.
    """
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        
        # Step 1: The discriminator consists of several convolutional layers that gradually reduce the image size.
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    apply_activation=True,
                    apply_normalization=False if idx == 0 else True,
                )
            )
            in_channels = feature

        # Step 2: After feature extraction, the output is passed through a fully connected layer to classify the image as real or fake.
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
