# Import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from MultiheadSelfAttentionBlock import MultiheadSelfAttentionBlock
from MLPBlock import MLPBlock

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the transformer Encoder block
# Encoder: Turn a sequence into learnable representation
# Decoder: Output of the encoder
# Residual connections: Add layer input to its subsequent output,this enables the creation of deeper networks.

# Create the class that inherits the nn.Module
class TransformerEncoderBlock(nn.Module):
  """Create a transformer Encoder block"""
  def __init__(self ,
               embedding_dim:int=768,
               num_heads:int=12,
               mlp_size:int=3072,
               mlp_dropout:float=0.1,
               attn_dropout:float=0):
    super().__init__()
    # Create MSA block
    self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)

    # Create MLP block
    self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                              mlp_size=mlp_size,
                              dropout=mlp_dropout)

  # create a forward method
  def forward(self , x):
    if isinstance(x , tuple):
      x = x[0]
    # Create residual connections for MSA block (add the input to the output)
    x = self.msa_block(x) + x
    # Create residual connections for MLP block(add the input to the output)
    x = self.mlp_block(x) + x
    return x
