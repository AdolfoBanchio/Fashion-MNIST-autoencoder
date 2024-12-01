import autoencoder
import autoencoder_trainer
import os
import json
import torch
from torchvision import datasets, transforms  
from autoencoder_trainer import AutoencoderTrainer, TrainingManager
from classifier import Classifier
from classifier_trainer import ClassifierTrainer
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

"""
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

train_set_autoencoder = autoencoder.CustomDataset(train_set_orig)
valid_set_autoencoder = autoencoder.CustomDataset(valid_set_orig)

""" 
trainer = TrainingManager(configurations, train_set_autoencoder, valid_set_autoencoder)

results = trainer.train_all() 
"""


# Entrenamiento del Clasificador con el mejor modelo de autoencoder (3)

# Se carga el mejor modelo de autoencoder
base_model = autoencoder.Autoencoder(dropout=0.2, l_size=1024)

best_model = autoencoder.Autoencoder(dropout=0.2, l_size=1024)
best_model.load_state_dict(torch.load("./results/model_3.pt", map_location=torch.device('cpu'), weights_only=True))

""" 
Entreno 3 clasificadores
1: Encoder sin pre-entrenamiento
2: Encoder con pre-entrenamiento, entreno todas las capas
3: Encoder con pre-entrenamiento, entreno solo la última capa
"""
classifiers = [
    #Classifier(autoencoder=best_model, num_classes=10),  # 3
    #Classifier(autoencoder=best_model, num_classes=10), # 2
    Classifier(autoencoder=base_model, num_classes=10) # 1
]

optimizers = [
    #torch.optim.Adam(classifiers[0].classifier.parameters(), lr=0.001),
    #torch.optim.Adam(classifiers[1].parameters(), lr=0.001),
    torch.optim.Adam(classifiers[0].parameters(), lr=0.001)
]

criterion = torch.nn.CrossEntropyLoss()

for i,(classifier, optimizer) in enumerate(zip(classifiers, optimizers)):
    # creamos los dataloaders
    train_loader = torch.utils.data.DataLoader(train_set_orig, batch_size=100, shuffle=True, num_workers=os.cpu_count()-1)
    valid_loader = torch.utils.data.DataLoader(valid_set_orig, batch_size=100, shuffle=True, num_workers=os.cpu_count()-1)

    trainer = ClassifierTrainer(classifier, train_loader, valid_loader, optimizer, criterion, epochs=60)
    trainer.train_model()

    # Guardamos los modelos y resultados
    torch.save(classifier.state_dict(), f"./results/classifier_{i+1}.pth")
    with open(f"./results/classifier_{i+1}_results.json", "w") as f:
        json.dump({
            "train_loss_incorrect": trainer.train_loss_incorrect,
            "train_loss": trainer.train_loss,
            "valid_loss": trainer.valid_loss,
            "train_precision_incorrect": trainer.train_precision_incorrect,
            "train_precision": trainer.train_precision,
            "valid_precision": trainer.valid_precision
        }, f)