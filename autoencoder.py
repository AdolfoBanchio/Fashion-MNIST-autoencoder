import os
import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose
import torchvision 
#from torchviz import make_dot
from tqdm import tqdm

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
    def __init__(self, dropout):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #nn.Flatten(),
            nn.Conv2d(1, 16, kernel_size=3, padding=0), # (1, 28, 28) -> (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2,2) # (16, 26, 26) -> (16, 13, 13)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*13*13, 16*13*13), # FULLY CONNECTED
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16,13,13)),
            nn.ConvTranspose2d(16,1,kernel_size=6, stride = 2, padding=1), # (16, 13, 13) -> (1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.decoder(x)
        return x

class Autoencoder_no_lineal(nn.Module):
    def __init__(self, dropout):
        super(Autoencoder_no_lineal, self).__init__()
        self.encoder = nn.Sequential(
            #nn.Flatten(),
            nn.Conv2d(1, 16, kernel_size=3, padding=0), # (1, 28, 28) -> (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2,2) # (16, 26, 26) -> (16, 13, 13)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,1,kernel_size=6, stride = 2, padding=1), # (16, 13, 13) -> (1, 28, 28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        x = self.decoder(x)
        return x

        
def train_one_epoch(model, train_loader, optimizer, criterion):
  model.train() # Se pone el modelo en modo de entrenamiento
  sum_batch_avg_loss = 0 # Inicializamos la suma de las pérdidas promedio de los batches

  for batch_number, (images, labels) in enumerate(train_loader):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      images = images.to(device) # Se envía la imagen al dispositivo
      labels = labels.to(device) # Se envía la etiqueta al dispositivo

      batch_size = len(images) # Se obtiene el tamaño del lote
      # Se obtiene la predicción del modelo y se calcula la pérdida 
      pred = model(images)
      loss = criterion(pred, labels)
      
      # Backpropagation usando el optimizador 
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      # Calculamos la perdida promedio del batch y lo agregamos a la suma total
      batch_avg_loss = loss.item() 
      sum_batch_avg_loss += batch_avg_loss
      
  # Calculamos la perdida promedio de todos los batches
  avg_loss = sum_batch_avg_loss / len(train_loader)
  # Calculamos la precisión del modelo
  return avg_loss


def validation_loop(model, valid_loader, criterion):
    model.eval() # Se pone el modelo en modo de evaluación

    sum_batch_avg_loss = 0 # Inicializamos la suma de las pérdidas promedio de los batches
    num_processed_examples = 0 # Inicializamos la cantidad de ejemplos procesados

    for batch_number, (images, labels) in enumerate(valid_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images = images.to(device) # Se envía la imagen al dispositivo
        labels = labels.to(device) # Se envía la etiqueta al dispositivo
      
        batch_size = len(images)

        # Se obtiene la predicción del modelo y se calcula la pérdida
        pred = model(images)
        loss = criterion(pred, labels)

        # Calculamos la perdida promedio del batch y lo agregamos a la suma total
        batch_avg_loss = loss.item()
        sum_batch_avg_loss += batch_avg_loss
        
        # Calculamos la cantidad total de predicciones procesadas
        num_processed_examples += batch_size

    # Calculamos la perdida promedio de todos los batches
    avg_loss = sum_batch_avg_loss / len(valid_loader)
    # Calculamos la precisión del modelo

    return avg_loss


def train_model(model, train_loader, valid_loader, optimizer, criterion, epochs):
  train_loss_incorrect = []
  train_loss = []
  
  valid_loss = []
  for epoch in tqdm(range(epochs)):

    # train one epoch
    train_entropy_inc = train_one_epoch(model, train_loader, optimizer, criterion)
    train_loss_incorrect.append(train_entropy_inc)

    # check avg loss and accuracy for incorrect predictions
    train_entropy = validation_loop(model, train_loader, criterion)
    train_loss.append(train_entropy)

    # validate the epoch
    valid_entropy = validation_loop(model, valid_loader, criterion)
    valid_loss.append(valid_entropy)

  return train_loss_incorrect, train_loss, valid_loss