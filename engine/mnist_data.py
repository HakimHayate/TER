from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from engine.data_augmentation import DataAugmentation

class MNIST_Data:
    """
    Gestionnaire de données pour le chargement et la préparation du dataset MNIST.
    
    Cette classe automatise le téléchargement, l'application de transformations 
    personnalisées (notamment l'ajout de bruit pour la robustesse) et la 
    création de générateurs de lots (DataLoaders).

    Attributes:
        enriched_dataset: Instance de DataAugmentation pour le prétraitement.
        train_data: Dataset MNIST d'entraînement avec bruit appliqué.
        test_data: Dataset MNIST de test avec bruit appliqué.
    """
    def __init__(self):
        """Initialise et télécharge les datasets avec les transformations PyTorch."""
        self.enriched_dataset = DataAugmentation()
        self.train_data = datasets.MNIST(root='data' , train=True,download=True, transform=transforms.Compose([transforms.ToTensor(),self.enriched_dataset.rectangle_noise],))
        self.test_data = datasets.MNIST(root='data' , train=False,download=True, transform=transforms.Compose([transforms.ToTensor(),self.enriched_dataset.rectangle_noise]))



    def get_loaders(self,batch_size,train = True):
        """
        Crée et retourne un DataLoader PyTorch.

        Args:
            batch_size (int): Nombre d'images par lot (ex: 64).
            train (bool): Si True, mélange les données (shuffle) pour l'entraînement. 
                          Si False, garde l'ordre pour le test.

        Returns:
            DataLoader: Itérateur sur les lots d'images et d'étiquettes.
        """
        if train:
            return DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        else:
            return DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        
