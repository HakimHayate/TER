import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import random
from engine import CNN
from engine.train import Trainer
import configuration as cf

class Finetuning(Dataset):
    """
    Dataset personnalisé pour charger les images réelles extraites des formulaires.
    
    Cette classe prépare les données (images de 28x28 pixels) pour l'entraînement 
    en les convertissant en tenseurs PyTorch normalisés [0, 1] et en ajoutant 
    la dimension du canal (Channel) nécessaire au CNN.

    Args:
        data_list (list): Liste de tuples (image_numpy, label_entier).
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        """Retourne le nombre total d'images prêtes pour le fine-tuning."""
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Récupère et formate un échantillon du dataset.
        Normalisation : division par 255.0.
        Transformation : ajout d'une dimension (1, 28, 28) pour PyTorch.
        """
        img, label = self.data_list[idx]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    """
    Pipeline de Fine-tuning et d'exportation vers ONNX.
    
    Déroulement :
    1. Chargement des images réelles triées par dossiers (étiquette dans le nom du fichier).
    2. Chargement des poids du modèle original MNIST (Transfer Learning).
    3. Fine-tuning : Ré-entraînement léger avec un taux d'apprentissage réduit (LR=0.0005).
    4. Exportation ONNX : Conversion du modèle dynamique PyTorch en graphe statique 
       optimisé pour l'inférence (compatible OpenVINO, ONNX Runtime).
    5. Configuration des axes dynamiques pour permettre des tailles de lots (batch) variables.
    """
    mydata = list()
    #finetuning que sur les 5 premières pages
    for page_idx in range(5):
        page_dir = os.path.join(cf.finetuning_data_path, str(page_idx))
        if not os.path.exists(page_dir): continue
        
        for img_name in os.listdir(page_dir):
            try:
                true_label = int(img_name.split("-")[-1].split(".")[0])
                img_path = os.path.join(page_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    if img.shape != (28, 28):
                        img = cv2.resize(img, (28, 28))
                    mydata.append((img, true_label))
            except (ValueError, IndexError):
                continue

    random.shuffle(mydata)
    dataset = Finetuning(mydata)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True)


    model = CNN()
    if os.path.exists(cf.model_path_pth):
        model.load_state_dict(torch.load(cf.model_path_pth))
        print("Poids du modèle original chargés.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) 
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, None, optimizer, criterion)
    
    print("Début du Fine-tuning...")
    trainer.train(epochs=10) 
    
    dummy_input = torch.randn(2, 1, 28, 28)
    model.eval()
    # 2. Exportation
    torch.onnx.export(
        model, 
        dummy_input, 
        "model_static_finetune.onnx", 
        export_params=True,       # Stocke les poids entraînés dans le fichier
        opset_version=15,         # Version stable et compatible
        do_constant_folding=True, # Optimise le modèle en simplifiant les constantes
        input_names=['input'],    # Nommer l'entrée (pratique pour OpenVINO/ONNX Runtime)
        output_names=['output'],  # Nommer la sortie
        dynamic_axes={'input' : {0 : 'batch_size'}, # Permet de changer la taille du batch (ex: 64)
                    'output' : {0 : 'batch_size'}}
    )
    print("Modèle exporté avec succès !")
    print(f"Modèle fine-tuné sauvegardé sous : model_static_finetune.onnx")