import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A convolutional block with optional batch normalization and activation functions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        apply_activation (bool): Applies an activation function if True.
    """
    def __init__(self, in_channels, out_channels, apply_activation, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True)
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if apply_activation
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.cnn(x))

    
class UpsampleBlock(nn.Module):
    """
    Upsampling block using PixelShuffle to increase resolution.
    
    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Factor by which the spatial resolution is increased.
    """
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(self.upsample(x)))


class DenseResidualBlock(nn.Module):
    """
    Residual block used in the generator to enhance feature extraction.
    
    Helps stabilize training by allowing gradients to flow through shortcut connections.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()
        
        for i in range(5):
            self.blocks.append(ConvBlock(
                in_channels + channels * i,
                channels if i <= 3 else in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                apply_activation=True if i <= 3 else False,
            ))

    def forward(self, x):
        new_inputs = x
        
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)

        return self.residual_beta * out + x


class RRDB(nn.Module):
    """
    A Residual-in-Residual Dense Block (RRDB) used in the ESRGAN generator to improve feature extraction.
    
    The RRDB consists of three DenseResidualBlocks to capture more complex patterns.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x


class Generator(nn.Module):
    """
    Generator model for ESRGAN to upscale low-resolution images to high resolution.
    
    Consists of an initial convolution, multiple RRDB blocks, feature refinement, upsampling layers, and final convolution layer to produce the output.
    
    Args:
        in_channels (int): Number of input channels (usually 3 for RGB).
        num_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of RRDB blocks.
    """
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super().__init__()
        
        # Step 1: Initial convolution to process the input.
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # Step 2: A stack of RRDB blocks to extract features.
        self.residuals = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])

        # Step 3: Final convolution layer to refine the features.
        self.convblock = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

        # Step 4: Upsampling layers to increase the resolution of the output.
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels), UpsampleBlock(num_channels))

        # Step 5: Final convolution to output the final image, with LeakyReLU for non-linearity.
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        initial = self.initial(x)
        x = self.conv(self.residuals(initial)) + initial
        x = self.upsamples(x)
        return self.final(x)


class Discriminator(nn.Module):
    """
    Discriminator model for ESRGAN to classify images as real or fake.
    
    Extracts features using convolutional layers and uses a fully connected layer to classify the image.
    
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
                    apply_activation=True,
                )
            )
            in_channels = feature

        # Step 2: After feature extraction, the output is passed through a fully connected layer to classify the image as real or fake.
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)


def initialize_weights(model, scale=0.1):
    """
    Initializes the weights of the model using Kaiming normal initialization.
    
    Args:
        model (nn.Module): The model to initialize.
        scale (float): The scaling factor for the weights.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale

        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale