import torch
import torch.nn as nn
import copy

class Classifier(nn.Module):
    def __init__(self, autoencoder, num_classes):
        super(Classifier, self).__init__()
        self.l_size = autoencoder.l_size
        self.dropout = autoencoder.dropout

        self.encoder = copy.deepcopy(autoencoder.encoder)
        self.classifier = nn.Sequential(
            nn.Linear(self.l_size, num_classes),
            nn.Sigmoid()
            )  # encoder[-3] es la capa lineal del encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x