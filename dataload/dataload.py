import os
import torch
from torchvision import datasets, transforms

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def get_dataloaders(data_dir, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    
    train_set = datasets.CIFAR100(
        root=os.path.join(data_dir, 'train'),
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_set = datasets.CIFAR100(
        root=os.path.join(data_dir, 'val'),
        train=False,
        download=True,
        transform=test_transform
    )
    
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.is_available() else 0
        ),
        'val': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0
        )
    }
    
    return dataloaders, train_set.classes
