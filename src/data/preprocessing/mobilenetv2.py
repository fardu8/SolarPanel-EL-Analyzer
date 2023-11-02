from src.data.dataset.elpv_dataset import ELPVDataset
from src.data.elpv_reader import load_dataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader


def get_processed_data_loaders():
    images, probs, types = load_dataset()
    labels = (probs * 3).astype(int)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ELPVDataset(images, labels, data_transforms)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(images) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader, dataset, train_dataset, val_dataset, test_dataset
