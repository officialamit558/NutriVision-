# Importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms


# Replicating the equation like MLP block
'''
MLP: A quite broad term for a layer wit a series of layer , layers can be multiple or even only one hidden layer.
Layer: Fully connected, dense , linear , feed-forward neural network layer.
GELU non linearity
Dropout: Regularization technique , applied every dense layer or fully connected layers
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating a class that inherits from nn.Module
class MLPBlock(nn.Module):
  # Create a layer normalized multilayer perceptron block
  def __init__(self,
               embedding_dim:int=768, # Hidden size D from table 1 for ViT-Base
               mlp_size:int=3072, # MLP size from table 1 ViT-Base
               dropout:float=0.1): # Dropout from table 3 for ViT-Base
    super().__init__()

    # Create the norm layer (LN)
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # create the multilayer perceptron (MLP) layer(s)
    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embedding_dim),
        nn.Dropout(p=dropout)
    )
  # Create a forward() method to pass the data through the layers
  def forward(self , x):
    x = self.layer_norm(x)
    x = self.mlp(x)
    return x