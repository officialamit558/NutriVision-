# Import the necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a class that inherits from nn.Module
class MultiheadSelfAttentionBlock(nn.Module):
  """
  Creates a multi-head self-attention block ("MSA" block for short)
  """
  # Initialize the class with hyperparameters from table 1
  def __init__(self,
               embedding_dim:int=768,
               num_heads:int=12,
               attn_dropout:float=0):
    super().__init__()
    # Create the norm layer (LN)
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # Create the multi-head attention (MSA) layer
    self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                num_heads=num_heads,
                                                dropout=attn_dropout,
                                                batch_first=True)

  # Create a forward() method to pass the data through the layers
  def forward(self , x):
    x = self.layer_norm(x)
    attn_output , _ = self.multihead_attn(query=x, # query embedding
                                      key=x,   # key embedding
                                      value=x, # value embedding
                                      need_weights=False)
    return attn_output # Here return only tensor , not return tuple
