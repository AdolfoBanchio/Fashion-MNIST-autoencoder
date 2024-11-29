# Configuración 0

```json
{
    "id": 1,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "l_size": 2304,
    "batch_size": 100,
    "epochs": 50,
    "lineal": true
}
```

![](./configuracion0.png)

Pérdida de entrenamiento incorrecto: 0.6816189906001091

Pérdida de entrenamiento: 0.6816189906001091

Pérdida de validación: 0.678600459098816

# Configuración 1

```json
{
    "id": 2,
    "learning_rate": 0.01,
    "dropout": 0.2,
    "l_size": 2304,
    "batch_size": 100,
    "epochs": 50,
    "lineal": true
}
```

![](./configuracion1.png)

Pérdida de entrenamiento incorrecto: 0.6816189914941788

Pérdida de entrenamiento: 0.6816189911961555

Pérdida de validación: 0.6786004507541656

# Configuración 2

```json
{
    "id": 3,
    "learning_rate": 0.01,
    "dropout": 0.2,
    "l_size": 2304,
    "batch_size": 100,
    "epochs": 50,
    "lineal": false
}
```

![](./configuracion2.png)

Pérdida de entrenamiento incorrecto: 0.5734905628363292

Pérdida de entrenamiento: 0.5773615228136381

Pérdida de validación: 0.5750147706270218

# Configuración 3

```json
{
    "id": 4,
    "learning_rate": 0.001,
    "dropout": 0.2,
    "l_size": 128,
    "batch_size": 100,
    "epochs": 50,
    "lineal": true
}
```

![](./configuracion3.png)

Pérdida de entrenamiento incorrecto: 0.6816189938783646

Pérdida de entrenamiento: 0.6816189893086751

Pérdida de validación: 0.6786004614830017

