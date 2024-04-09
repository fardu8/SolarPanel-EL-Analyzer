import os

import torch

from src.data.dataset.elpv_dataset import ELPVDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
import joblib as joblib

from src.data.elpv_reader import load_dataset


def get_processed_data_loaders(type, load_data=False):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    train_loader_filename = os.path.join(root_dir, 'data_loaders', type, 'train_loader.pkl')
    val_loader_filename = os.path.join(root_dir, 'data_loaders', type, 'val_loader.pkl')
    test_loader_filename = os.path.join(root_dir, 'data_loaders', type, 'test_loader.pkl')
    train_dataset_filename = os.path.join(root_dir, 'data_loaders', type, 'train_dataset.pkl')
    val_dataset_filename = os.path.join(root_dir, 'data_loaders', type, 'val_dataset.pkl')
    if load_data and os.path.exists(train_loader_filename) and os.path.exists(val_loader_filename) and os.path.exists(test_loader_filename) and os.path.exists(train_dataset_filename) and os.path.exists(val_dataset_filename):
        train_loader = joblib.load(train_loader_filename)
        val_loader = joblib.load(val_loader_filename)
        train_dataset = joblib.load(train_dataset_filename)
        val_dataset = joblib.load(val_dataset_filename)
        test_loader = joblib.load(test_loader_filename)
        print('data loaders loaded')
        return train_loader, val_loader, test_loader, train_dataset, val_dataset

    images, probs, types = load_dataset()
    labels = (probs * 3).astype(int)
    if type != 'both':
        filtered_indices = np.where(types == type)[0]
        images = images[filtered_indices]
        labels = labels[filtered_indices]

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ELPVDataset(images, labels, data_transforms)
    torch.manual_seed(42)

    train_size = int(0.65 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(images) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    joblib.dump(train_loader, train_loader_filename)
    joblib.dump(val_loader, val_loader_filename)
    joblib.dump(train_dataset, train_dataset_filename)
    joblib.dump(val_dataset, val_dataset_filename)
    joblib.dump(test_loader, test_loader_filename)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset
