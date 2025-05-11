# Import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a class called PatchEmbedding
class PatchEmbedding(nn.Module):
  # Initialize of the layer with appropriate hyperparameters
  def __init__(self,
               in_channels:int=3,
               patch_size:int=16,
               embedding_dim:int=768):
    super().__init__()
    self.patch_size = patch_size
    # create a layer to turn an image into embedded patches
    self.patcher = nn.Conv2d(
        in_channels=in_channels,
        out_channels=embedding_dim,
        kernel_size=patch_size,
        stride=patch_size,
        padding=0
    )
    # Create a layer to flatten feature map output of conv2d
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)

  # Define a forward method to define the forward computation shape
  def forward(self , x):
    # create assertin to check that inputs are the corrects shape
    # image_resolution = x.shape[-1]
    # assert image_resolution % patch_size == 0

    # Perform the forward pass
    x_patched = self.patcher(x)
    x_flattened = self.flatten(x_patched)
    return x_flattened.permute(0 , 2 , 1)