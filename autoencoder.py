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
        self.dropout = dropout
        self.l_size = l_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (1, 28, 28) -> (32, 28, 28)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 28, 28) -> (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 14, 14) -> (64, 14, 14)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2) , # (64, 14, 14) -> (64, 7, 7)
            # capa lineal
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, l_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(l_size, 64 * 7 * 7),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (64, 7, 7)),
            # 1 capa convolucional
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (64, 7, 7) -> (32, 14, 14)
            nn.ReLU(),
            nn.Dropout(dropout),
            # 2 capa convolucional
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (32, 14, 14) -> (1, 28, 28)
            nn.Sigmoid()  # Normalización a [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder_no_lineal(nn.Module):
    def __init__(self, dropout):
        super(Autoencoder_no_lineal, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (1, 28, 28) -> (32, 28, 28)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 28, 28) -> (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 14, 14) -> (64, 14, 14)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2) , # (64, 14, 14) -> (64, 7, 7)
        )
        # Decoder
        self.decoder = nn.Sequential(
            # 1 capa convolucional
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (64, 7, 7) -> (32, 14, 14)
            nn.ReLU(),
            nn.Dropout(dropout),
            # 2 capa convolucional
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),  # (32, 14, 14) -> (1, 28, 28)
            nn.Sigmoid()  # Normalización a [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x