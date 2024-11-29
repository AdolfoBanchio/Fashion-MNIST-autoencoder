import autoencoder
import training
import os
import json
import torch
from torchvision import datasets, transforms

def train_configuration(config, t_dataset, v_dataset):
    """ 
    Entrena y valida un autoencoder con los datos de t_dataset y v_dataset

    La función recibe un diccionario con la configuración del autoencoder de la forma:
    {
        "learning_rate": float,
        "dropout": float,
        "batch_size": int,
        "epochs": int,
        "lineal": bool True si el modelo tiene capa lienal intermedia, False c.c. 
    } 
    """
    lr = config["learning_rate"]
    dropout = config["dropout"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Se crea el autoencoder
    if not config["lineal"]:
        model = autoencoder.Autoencoder_no_lineal(dropout)
    else:
        model = autoencoder.Autoencoder(dropout)

    # Se envía el modelo al dispositivo
    model.to(device)

    # Se crea el optimizador y la función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Se crea el DataLoader para los datos de entrenamiento y validación
    train_loader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)
    valid_loader = torch.utils.data.DataLoader(v_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()-1)

    # Se entrena el modelo
    train_loss_incorrect, train_loss, valid_loss = training.train_model(model, train_loader, valid_loader, optimizer, criterion, epochs)

    return model, train_loss_incorrect, train_loss, valid_loss


def train_all(configs, t_dataset, v_dataset):
    """
    Entrena y valida todos los autoencoders con las configuraciones en configs

    La función recibe una lista de diccionarios con las configuraciones de los autoencoders de la forma:
    [
        {
            "name": str,
            "learning_rate": float,
            "dropout": float,
            "batch_size": int,
            "epochs": int,
            "lineal": bool True si el modelo tiene capa lienal intermedia, False c.c. 
        },
        ...
    ] 
    Guarda los resultados en archivos .json y los modelos en archivos .pt
    """
    results = {}
    for i,(config) in enumerate(configs):
        model, train_loss_incorrect, train_loss, valid_loss = train_configuration(config, t_dataset, v_dataset)
        id = config["id"]
        results = {
            f"config_{id}": config,
            "train_loss_incorrect_"+str(id): train_loss_incorrect,
            "train_loss_"+str(id): train_loss,
            "valid_loss_"+str(id): valid_loss
        }
        # save the state dict of the model
        torch.save(model.state_dict(), f'./results/model_{id}.pt')
        with open(f'./results/result_{id}.json', 'w') as f:
            json.dump(results, f)
            
configurations = [
    {
        "id": 1,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 16*12*12, # full connected desp del encoder
        "batch_size": 100,
        "epochs": 50,
    },
    {
        "id": 2,
        "learning_rate": 0.01,
        "dropout": 0.2,
        "l_size": 16*12*12, # full connected desp del encoder
        "batch_size": 100,
        "epochs": 50,
    },
    {
        "id": 3,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "l_size": 128,
        "batch_size": 100,
        "epochs": 50,
    },
]

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

# Se cargan los datasets
train_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = True,  transform = transform)
valid_set_orig = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)

train_set_autoencoder = autoencoder.CustomDataset(train_set_orig)
valid_set_autoencoder = autoencoder.CustomDataset(valid_set_orig)

train_all(configurations, train_set_autoencoder, valid_set_autoencoder)
