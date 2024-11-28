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
        results[f"config_{i}"] = (model, train_loss_incorrect, train_loss, valid_loss)
    return results

def save_results(results):
    """  
    Guarda un archivo JSON con los resultados en path

    Recive un diccionario de diccionarios con los resultados de la forma:
    {
        "config_0": (modelo_0, train_loss_incorrect_0, train_loss_0, valid_loss_0),
        "config_1": (modelo_1, train_loss_incorrect_1, train_loss_1, valid_loss_1),
    """

    for result in results:
        to_save = {
            "model": results[result][0].state_dict(),
            "train_loss_incorrect": results[result][1],
            "train_loss": results[result][2],
            "valid_loss": results[result][3]
        }
        with open(f'./results/{result}.json', 'w') as f:
            json.dump(to_save, f)


configurations = [
    { # configuracion default
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 100,
        "epochs": 50,
        "lineal": True
    },
    {
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs": 50,
        "lineal": True
    },
    {
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 16,
        "epochs": 50,
        "lineal": True
    },
    {
        "learning_rate": 0.001,
        "dropout": 0.2,
        "batch_size": 64,
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

results = train_all(configurations, train_set_autoencoder, valid_set_autoencoder)
save_results(results)