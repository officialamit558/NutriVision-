# Importing the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from re import X
from PatchEmbedding import PatchEmbedding
from MultiheadSelfAttentionBlock import MultiheadSelfAttentionBlock
from TransformerEncoderBlock import TransformerEncoderBlock

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Putting all above method together and make the end to end ViT model
class ViT(nn.Module):
  def __init__(self,
               img_size:int=224,
               in_channels:int=3,
               patch_size:int=16,
               num_transfomer_layer:int=12,
               embedding_dim:int=768,
               mlp_size:int=3072,
               num_heads:int=12,
               attn_dropout:float=0,
               mlp_dropout:float=0.1,
               embedding_dropout:float=0.1,
               num_classes:int=1000):
    super().__init__()
    # make an assertion that the image size is compatible with the size
    assert img_size % patch_size == 0 , f"Image size must be divisible by patch size , image size: {img_size}"

    # calculate the number of the patches (height * width) // patch_size **2
    self.num_patches = (img_size * img_size) // patch_size ** 2

    # Create learnable class embedding
    self.class_embedding = nn.Parameter(torch.ones(1 , 1, embedding_dim),requires_grad=True)

    # Create learnable position embedding
    self.position_embedding = nn.Parameter(torch.ones(1 , self.num_patches + 1 , embedding_dim) , requires_grad=True)

    # create embedding dropout value
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    # Create patch embedding layer
    self.patch_embedding_layer = PatchEmbedding(in_channels=in_channels,
                                                patch_size=patch_size,
                                                embedding_dim=embedding_dim)

    # Creating the transformer encoder block
    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,num_heads=num_heads , mlp_size=mlp_size,mlp_dropout=mlp_dropout) for _ in range(num_transfomer_layer)])

    # Create the classification head
    self.classification_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self , x):
    # Get the batch size
    batch_size = x.shape[0]
    # Create class token embedding and expand it to match the batch size
    class_token = self.class_embedding.expand(batch_size , -1, -1)
    # Create the patch embedding
    x = self.patch_embedding_layer(x)
    # concat class token embedding and patch embedding
    x = torch.cat((class_token , x) , dim=1)
    # Add the position embedding with class and patch embedding
    x = self.position_embedding + x
    # Apply dropout to patch embedding
    x = self.embedding_dropout(x)
    # pass position and patch embedding to transform encoder
    x = self.transformer_encoder(x)
    # put 0th index logit through the classifier
    x = self.classification_head(x[:,0])
    return x
