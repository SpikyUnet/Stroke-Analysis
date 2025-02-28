import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            x, y = self.transform(x), self.transform(y)
        return x, y

def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size=5):
    train_transform = T.Compose([T.RandomRotation(10)])
    augmented_train_dataset = CustomDataset(X_train, Y_train, transform=train_transform)
    original_train_dataset = CustomDataset(X_train, Y_train)
    train_dataset = torch.utils.data.ConcatDataset([original_train_dataset, augmented_train_dataset])
    
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(CustomDataset(X_val, Y_val), batch_size=batch_size, shuffle=False),
        "test": DataLoader(CustomDataset(X_test, Y_test), batch_size=1, shuffle=False)
    }
