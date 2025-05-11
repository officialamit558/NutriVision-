from pathlib import Path
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from TransferLearning import create_ViT_model
# Make sure to transform the images to the same size as the model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

food_101_transform = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    transform
])

# Load trasform
vit_101 , vit_101_transform = create_ViT_model(classes=101,
                               seed=42)

data_dir = Path('data')
train_data = datasets.Food101(
    root=data_dir,
    split='train',
    transform=vit_101_transform,
    download=True
)
test_data = datasets.Food101(
    root=data_dir,
    split='test',
    transform=vit_101_transform,
    download=True
)

# Create DataLoader for training and testing datasets
BATCH_SIZE = 32
train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# print(f"Number of training samples: {len(train_data)}")
# print(f"Number of testing samples: {len(test_data)}")
# print(f"Number of classes: {len(train_data.classes)}")