import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import autoencoder

class ClassifierTrainer:
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

        self.train_precision_incorrect = []
        self.train_precision = []
        self.valid_precision = []

    def train_loop(self):
        self.model.train()
        
        sum_correct = 0 # Inicializamos la suma de las predicciones correctas
        num_processed_examples = 0 # Inicializamos la cantidad de ejemplos procesados
        running_loss = 0

        for batch_number, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            batch_size = len(images)

            pred = self.model(images)
            loss = self.criterion(pred, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            sum_correct += (pred.argmax(1) == labels).sum().item()

            num_processed_examples += batch_size

        avg_loss = running_loss / len(self.train_loader)

        avg_precision = sum_correct / len(self.train_loader.dataset)
        return avg_loss, avg_precision

    def validation_loop(self, loader):
        self.model.eval()

        sum_correct = 0 # Inicializamos la suma de las predicciones correctas
        num_processed_examples = 0 # Inicializamos la cantidad de ejemplos procesados
        running_loss = 0
        
        with torch.no_grad():
            for X, Y in loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, Y)

                running_loss += loss.item()
                
                sum_correct += (pred.argmax(1) == Y).sum().item()
                num_processed_examples += len(X)

        avg_loss = running_loss / len(loader)
        avg_precision = sum_correct / len(loader.dataset)
        return avg_loss, avg_precision

    def train_model(self):
        for _ in tqdm(range(self.epochs)):
            train_loss_inc, train_precision_inc = self.train_loop()
            self.train_loss_incorrect.append(train_loss_inc)
            self.train_precision_incorrect.append(train_precision_inc)

            train_loss, train_precision = self.validation_loop(self.train_loader)
            self.train_loss.append(train_loss)
            self.train_precision.append(train_precision)

            valid_loss, valid_precision = self.validation_loop(self.valid_loader)
            self.valid_loss.append(valid_loss)
            self.valid_precision.append(valid_precision)

        return self.train_loss_incorrect, self.train_loss, self.valid_loss, self.train_precision_incorrect, self.train_precision, self.valid_precision