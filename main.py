import autoencoder
import autoencoder_trainer
import os
import json
import torch
from torchvision import datasets, transforms  
from autoencoder_trainer import AutoencoderTrainer, TrainingManager
 
""" 
    {
        "id": 1,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 64,  # Latent size más pequeño
        "batch_size": 100,
        "epochs": 60,
        "lineal": True
    },
    {
        "id": 2,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 512,  # Latent size intermedio
        "batch_size": 100,
        "epochs": 60,
        "lineal": True
    },
    {
        "id": 4,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 128,  # no importa el latent size
        "batch_size": 100,
        "epochs": 60,
        "lineal": False
    }
"""

configurations = [
    {
        "id": 3,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 1024,  # Latent size más grande
        "batch_size": 100,
        "epochs": 60,
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