import autoencoder
import training
import os
import json
import torch
from torchvision import datasets, transforms  
from training import AutoencoderTrainer, TrainingManager
 
configurations = [
    {
        "id": 1,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 256, 
        "batch_size": 100,
        "epochs": 5,
        "lineal": False
    },
    {
        "id": 2,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 256, 
        "batch_size": 100,
        "epochs": 5,
        "lineal": True
    },
]

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

train_set_autoencoder = autoencoder.CustomDataset(train_set_orig)
valid_set_autoencoder = autoencoder.CustomDataset(valid_set_orig)

trainer = TrainingManager(configurations, train_set_autoencoder, valid_set_autoencoder)

results = trainer.train_all()