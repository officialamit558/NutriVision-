# import libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from ViT import ViT
from dataset import train_dataloader , test_dataloader
from TransferLearning import create_ViT_model

from NeuroPipe.NeuroLoom import (
    train,
    save_model,
    plot_loss_curves,
    set_seeds,
    save_model
)

if __name__ == "__main__":
    # Set the seed
    set_seeds()

    # Make the instances of the ViT model of 101 classes
    vit_101 , vit_101_transform = create_ViT_model(classes=101,
                               seed=42)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the optimizer and loss_function
    optimizer = torch.optim.Adam(
        vit_101.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.1
    )
    # Set the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Now time to train the model on the Food101 dataset

    results_food101_vit_model = train(
        model=vit_101,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device
    )

    plot_loss_curves(results_food101_vit_model)
    save_model(
        model=vit_101,
        target_dir="models",
        model_name="food101_vit_model.pth"
    )