import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

mean = (0.4914, 0.4822, 0.4465)
std = (0.247, 0.243, 0.261)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Random crop
    transforms.RandomHorizontalFlip(),     # Horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
    transforms.RandomRotation(15),         # Random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

data_train = CIFAR10("./data/cifar10", download=True, train=True, transform=transform_train)
data_val = CIFAR10("./data/cifar10", download=False, train=False, transform=transform_test)

dataloader_train = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
dataloader_val = DataLoader(data_val, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}
