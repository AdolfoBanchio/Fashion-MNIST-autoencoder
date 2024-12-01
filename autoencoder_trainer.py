import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import autoencoder

class AutoencoderTrainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, epochs):
      self.model = model
      self.train_loader = train_loader
      self.valid_loader = valid_loader
      self.optimizer = optimizer
      self.criterion = criterion
      self.epochs = epochs
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self.train_loss_incorrect = []
      self.train_loss = []
      self.valid_loss = []

    def train_loop(self):
      self.model.train()
      running_loss = 0

      for batch_number, (images, labels) in enumerate(self.train_loader):

        images = images.to(self.device)
        labels = labels.to(self.device)

        pred = self.model(images)
        loss = self.criterion(pred, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        
      avg_loss = running_loss / len(self.train_loader)
      return avg_loss


    def validation_loop(self, loader):
      self.model.eval()
      running_loss = 0
      num_batches = len(loader)

      with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, Y)

                running_loss += loss.item()

      avg_loss = running_loss / num_batches
      return avg_loss


    def train_model(self):
      for _ in tqdm(range(self.epochs)):
        self.train_loss_incorrect.append(self.train_loop())
        self.train_loss.append(self.validation_loop(self.train_loader))
        self.valid_loss.append(self.validation_loop(self.valid_loader))

      return self.train_loss_incorrect, self.train_loss, self.valid_loss
    
  
class TrainingManager:
    def __init__(self, configs, t_dataset, v_dataset):
      self.configs = configs
      self.t_dataset = t_dataset
      self.v_dataset = v_dataset

    def train_configuration(self, config):
      lr = config["learning_rate"]
      dropout = config["dropout"]
      l_size = config["l_size"]
      batch_size = config["batch_size"]
      epochs = config["epochs"]

      if not config["lineal"]:
        model = autoencoder.Autoencoder_no_lineal(dropout)
      else:
        model = autoencoder.Autoencoder(dropout, l_size=l_size)

      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
      criterion = torch.nn.MSELoss()

      train_loader = torch.utils.data.DataLoader(self.t_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)
      valid_loader = torch.utils.data.DataLoader(self.v_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)


      trainer = AutoencoderTrainer(model, train_loader, valid_loader, optimizer, criterion, epochs)

      train_loss_incorrect, train_loss, valid_loss = trainer.train_model()

      return model, train_loss_incorrect, train_loss, valid_loss
    
    def train_all(self):
      results = {}
      for config in self.configs:
        model, train_loss_incorrect, train_loss, valid_loss = self.train_configuration(config)
        id = config["id"]
        results = {
          f"config": config,
          "train_loss_incorrect": train_loss_incorrect,
          "train_loss": train_loss,
          "valid_loss": valid_loss
        }
        torch.save(model.state_dict(), f'./results/model_{id}.pt')
        with open(f'./results/result_{id}.json', 'w') as f:
          json.dump(results, f)