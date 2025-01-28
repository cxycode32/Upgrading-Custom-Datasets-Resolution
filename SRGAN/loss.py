import config
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    """
    A perceptual loss based on the VGG19 network. It measures the similarity 
    between generated and target images by comparing their high-level features 
    from a pretrained VGG19 network.

    Used in SRGAN to help the generator produce perceptually better images.

    Attributes:
        vgg (nn.Module): A frozen VGG19 feature extractor (first 36 layers).
        loss (nn.MSELoss): Mean Squared Error loss function to compute the feature map differences.
    """
    
    def __init__(self):
        """
        Initializes the VGGLoss module.

        - Loads a pretrained VGG19 network and extracts features up to layer 36.
        - Freezes the network to prevent gradient updates.
        - Defines MSELoss to compare extracted features.
        """
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36]
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.vgg = self.vgg.eval().to(config.DEVICE)
        self.loss = nn.MSELoss()


    def forward(self, input, target):
        """
        Computes the perceptual loss between input and target images.

        Args:
            input (torch.Tensor): The generated image tensor.
            target (torch.Tensor): The ground truth high-resolution image tensor.

        Returns:
            torch.Tensor: The MSE loss between VGG feature maps of input and target images.
        """
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


