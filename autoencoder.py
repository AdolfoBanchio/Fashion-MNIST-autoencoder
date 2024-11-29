import torch 
from torch import nn
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, image


class Autoencoder(nn.Module):
    def __init__(self, dropout, l_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0),  # (1, 28, 28) -> (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),  # (16, 26, 26) -> (32, 24, 24)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2),  # (32, 24, 24) -> (32, 12, 12)
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, l_size),  # FULLY CONNECTED
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(l_size, 32 * 12 * 12),  # FULLY CONNECTED
            nn.ReLU(),
            nn.Unflatten(1, (32, 12, 12)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (32, 12, 12) -> (16, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=0),  # (16, 24, 24) -> (1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder_no_lineal(nn.Module):
    def __init__(self, dropout):
        super(Autoencoder_no_lineal, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0), # (1, 28, 28) -> (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=0), # (16, 26, 26) -> (16, 24, 24)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2,2) # (32, 24, 24) -> (32, 12, 12)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0), # (16, 12, 12) -> (16, 24, 24)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1, padding=0),  # (16, 24, 24) -> (1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x