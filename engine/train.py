import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from engine import CNN
from engine.mnist_data import MNIST_Data
import time
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import random
import os
import numpy as np

def seed_everything(seed=42):
    """
    Fixe les graines de calcul (seeds) pour garantir la reproductibilité totale.
    
    Configure les générateurs de nombres aléatoires de Python, NumPy et PyTorch, 
    et force les algorithmes de convolution de CuDNN à être déterministes 
    pour éliminer toute variabilité entre deux exécutions.

    Args:
        seed (int): Valeur de la graine utilisée pour initialiser les générateurs (défaut 42).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class Trainer:
    """
    Gestionnaire d'entraînement et d'évaluation pour les modèles de Deep Learning.
    
    Cette classe encapsule la boucle d'optimisation, le calcul des pertes, 
    et la génération de métriques de performance (Accuracy, Matrice de confusion).

    Attributes:
        model: Le réseau de neurones à entraîner (nn.Module).
        train_loader: DataLoader contenant les données d'entraînement.
        test_loader: DataLoader contenant les données de validation/test.
        optimizer: Algorithme d'optimisation (ex: Adam, SGD).
        criterion: Fonction de perte (ex: CrossEntropyLoss).
    """
    def __init__(self, model,train_loader,test_loader,optimizer,criterion) :
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader  
        self.optimizer = optimizer
        self.criterion = criterion
        self.all_preds = list()
        self.all_labels = list()
        self.conf_matrix = None
        self.losses = list()

    def train(self,epochs):
        """
        Exécute la boucle d'entraînement sur plusieurs époques.
        
        Réalise le passage avant (forward), le calcul de l'erreur, la rétropropagation 
        du gradient (backward) et la mise à jour des poids de l'optimiseur.

        Args:
            epochs (int): Nombre de passages complets sur le jeu de données.
        """
        for epoch in range(epochs) :
            self.model.train()
            for batch_idx, (x_train,y_train) in enumerate(self.train_loader) :
                y_pred = self.model(x_train)
                loss = self.criterion(y_pred,y_train)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.losses.append(loss.item())
                if batch_idx % 100 == 0 :
                    print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item()}")

    def test(self) :
        """
        Évalue le modèle sur le jeu de test sans mise à jour des gradients.
        
        Désactive le dropout et la batch normalization (mode eval) pour collecter 
        les prédictions et calculer la matrice de confusion finale.
        """
        self.model.eval()
        with torch.no_grad() :
            for x_test,y_test in self.test_loader :
                y_pred = self.model(x_test)
                self.all_preds.append(y_pred.argmax(dim=1))
                self.all_labels.append(y_test)
        self.all_preds = torch.cat(self.all_preds)
        self.all_labels = torch.cat(self.all_labels)
        self.conf_matrix = confusion_matrix(self.all_labels.cpu(),self.all_preds.cpu())

    def get_confusion_matrix(self) :
        """Retourne la matrice de confusion calculée lors du dernier test."""
        return self.conf_matrix
    
    def get_accuracy(self) :
        """Calcule le score de précision global (accuracy) entre 0 et 1."""
        return (self.all_preds == self.all_labels).sum().item() / len(self.all_labels)


if __name__ == "__main__":
    """
    Point d'entrée principal du script d'entraînement et d'évaluation.
    
    Déroulement du pipeline :
    1. Initialisation des données MNIST et des DataLoaders.
    2. Instanciation du modèle CNN, de l'optimiseur (Adam) et de la fonction de perte.
    3. Exécution de la boucle d'entraînement (Train) et mesure du temps de calcul.
    4. Évaluation sur le jeu de test et génération des métriques (Accuracy).
    5. Visualisation de la courbe de perte (Loss) et de la matrice de confusion.
    6. Sauvegarde des poids du modèle entraîné au format .pth.
    """
    data = MNIST_Data()
    train_loader = data.get_loaders(batch_size=64,train=True)
    test_loader = data.get_loaders(batch_size=64,train=False)
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model,train_loader,test_loader,optimizer,criterion)
    start = time.time()
    trainer.train(epochs=1)
    end = time.time()
    trainer.test()
    step = 50   # affiche 1 loss sur 50 
    plt.plot(trainer.losses[::step])
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

    confusion_matrix = trainer.get_confusion_matrix()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    disp.plot(cmap="Blues", values_format="d")  
    plt.title("Confusion Matrix")
    plt.show()
    print(f"Training time : {(end - start) / 60} minutes")
    print(f"Accuracy : {trainer.get_accuracy()}")
    torch.save(model.state_dict(),"model.pth")

    model.eval() # Très important : fige les couches BatchNorm et Dropout
    
    dummy_input = torch.randn(2, 1, 28, 28) 
    
    # Exportation ONNX
    onnx_path = "model_static.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,        # Inclut les poids entraînés
        opset_version=15,          # Version stable standard
        do_constant_folding=True,  # Optimisation du graphe
        input_names=['input'],     # Nom de l'entrée
        output_names=['output'],    # Nom de la sortie
        dynamic_axes={'input' : {0 : 'batch_size'}, # Permet de varier la taille du batch en inférence
                      'output' : {0 : 'batch_size'}}
    )
    
    print(f"Modèle ONNX sauvegardé sous : {onnx_path}")

