import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Architecture de réseau de neurones convolutif (CNN) optimisée pour la classification MNIST.
    
    Le modèle est structuré en trois blocs de convolution profonds, utilisant 
    la Batch Normalization pour la stabilité et le Dropout pour la régularisation. 
    Il se termine par une couche Global Average Pooling (GAP) qui réduit drastiquement 
    le nombre de paramètres par rapport à une couche dense classique.

    Structure :
        - Bloc 1 & 2 : Double convolution + Batch Norm + MaxPool + Dropout.
        - Bloc 3 : Double convolution + Batch Norm pour l'extraction de caractéristiques hautes.
        - Tête de classification : Global Average Pooling + Fully Connected layers.
    """
    def __init__(self, num_classes=10):

        super().__init__()

        # Bloc 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        # Bloc 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        # Bloc 3
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)


        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)


        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        """
        Définit le flux de données à travers les couches du réseau (Passage avant).
        
        Args:
            x (torch.Tensor): Tenseur d'entrée de forme (Batch, 1, 28, 28).

        Returns:
            torch.Tensor: Logits de sortie pour les 10 classes (chiffres de 0 à 9).
        """
        # Bloc 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Bloc 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Bloc 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        # GAP
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)

        return x