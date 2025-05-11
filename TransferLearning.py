# Import libraries
import os
import torch
import torch.nn as nn
import torchvision

# Create a vit feature extractor function
def create_ViT_model(classes:int=3,
                     seed:int=42):
  # load the weight of vit__16
  weight = torchvision.models.ViT_B_16_Weights.DEFAULT
  transforms = weight.transforms()
  model = torchvision.models.vit_b_16(weights=weight)

  # freeze the model
  for param in model.parameters():
    param.requires_grad = False

  # change the last layer of the vit model
  torch.manual_seed(seed)
  model.heads = torch.nn.Sequential(
      torch.nn.Linear(in_features=768,out_features=classes)
  )

  return model , transforms