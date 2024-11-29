import autoencoder
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
    train_loss_incorrect, train_loss, valid_loss = autoencoder.train_model(model, train_loader, valid_loader, optimizer, criterion, epochs)

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

    Retorna una lista de tuplas con la forma:
    [
        (modelo_1, train_loss_incorrect_1, train_loss_1, valid_loss_1),
        (modelo_2, train_loss_incorrect_2, train_loss_2, valid_loss_2),
        ...
    ]
    """
    results = {}
    for i,(config) in enumerate(configs):
        model, train_loss_incorrect, train_loss, valid_loss = train_configuration(config, t_dataset, v_dataset)
        results = {
            config["name"]: config,
            "train_loss_incorrect_"+str(i): train_loss_incorrect,
            "train_loss_"+str(i): train_loss,
            "valid_loss_"+str(i): valid_loss
        }
        # save the state dict of the model
        torch.save(model.state_dict(), f'./results/model_{i}.pt')
        with open(f'./results/result_{i}.json', 'w') as f:
            json.dump(results, f)
            

configurations = [
    { # configuracion default
        "id": 0,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 50,
        "lineal": True
    },
    {
        "id": 1,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs": 50,
        "lineal": True
    },
    {
        "id": 2,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 16,
        "epochs": 50,
        "lineal": True
    },
    {
        "id": 3,
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 50,
        "lineal": False
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
