from ViT import ViT
from torchinfo import summary
import os
from TransferLearning import create_ViT_model
# Set the instances the ViT model class
vit_101 , vit_transform = create_ViT_model(classes=101,
                           seed=42)

summary(model=vit_101,
        input_size=(1, 3, 224, 224),
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])