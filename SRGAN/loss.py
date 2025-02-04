import config
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    """
    A perceptual loss based on the VGG19 network. It measures the similarity 
    between generated and target images by comparing their high-level features 
    from a pretrained VGG19 network.

    This loss is commonly used in SRGANs to guide the generator in producing 
    perceptually realistic images by minimizing the difference in feature representations.

    Attributes:
        vgg (nn.Module): A feature extractor using the first 36 layers of a pretrained VGG19 
            network. It is frozen (requires_grad=False) and set to evaluation mode.
        loss (nn.MSELoss): Mean Squared Error loss function used to compare the extracted 
            feature maps from the input and target images.
    """
    
    def __init__(self):
        """
        Initializes the VGGLoss module.

        - Loads a pretrained VGG19 network and extracts features up to layer 36.
        - Freezes the network to prevent gradient updates, ensuring it remains a fixed feature extractor.
        - Sets the network to evaluation mode to maintain consistency in behavior.
        - Defines MSELoss to compare the extracted feature representations of input and target images.
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

        This function extracts high-level features from both the generated (input) 
        and ground truth (target) images using the frozen VGG19 feature extractor. 
        The difference between these feature representations is computed using MSE loss.

        Args:
            input (torch.Tensor): The generated image tensor.
            target (torch.Tensor): The ground truth high-resolution image tensor.

        Returns:
            torch.Tensor: The computed MSE loss between VGG feature maps of input and target images.
        """
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)
