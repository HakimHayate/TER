from engine.model import CNN
import torch
import configuration as cf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

class Identification:
    """
    Moteur d'inférence haute performance pour l'identification des caractères.
    
    Cette classe utilise ONNX Runtime pour exécuter le modèle de classification 
    de manière optimisée. Elle gère le prétraitement des lots (batch), 
    la prédiction vectorisée et le calcul des métriques de performance finales 
    (précision et matrice de confusion).

    Attributes:
        session (ort.InferenceSession): Session ONNX chargée pour l'inférence rapide.
        input_name (str): Nom du point d'entrée du modèle ONNX.
        accuracy (float): Score de précision calculé après identification.
        cm_norms (numpy.ndarray): Matrice de confusion normalisée.
        cm_counts (numpy.ndarray): Matrice de confusion en valeurs absolues.
    """
    def __init__(self,model_path=cf.model_path):
        self.model = CNN()
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.identify_path = cf.identify_path
        self.cm_norms = None
        self.accuracy = None
        self.cm_counts = None
        

    def predict(self, imgs):
        """
        Réalise la prédiction sur un lot d'images.
        
        Prépare les tableaux NumPy (conversion float32, normalisation [0, 1], 
        ajout de la dimension canal) et exécute le modèle via ONNX Runtime.

        Args:
            imgs (list): Liste d'images (numpy arrays ou tenseurs).

        Returns:
            numpy.ndarray: Liste des classes prédites (chiffres de 0 à 9).
        """
        processed_imgs = []
            
        for img in imgs:
                
            if hasattr(img, 'numpy'):
                img = img.detach().cpu().numpy()

            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img /= 255.0

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
                
            processed_imgs.append(img)

        batch_tensor = np.stack(processed_imgs)

        outputs = self.session.run(None, {self.input_name: batch_tensor})
            
        return np.argmax(outputs[0], axis=1)

    def fast_identify(self, batch_generateur):
        """
        Traite un flux de données complet pour une identification globale.
        
        Parcourt le générateur de lots, extrait les étiquettes réelles depuis 
        les identifiants de cases, et agrège les résultats pour calculer 
        les statistiques de performance.

        Args:
            batch_generateur (generator): Flux de lots (image, index) venant du préprocess.
        """
        y_true = []
        y_pred = []

        for batch in batch_generateur:

                imgs = []
                true_labels = []

                for crop, crop_idx in batch:
                    try:
                        true_label = int(crop_idx.split("-")[-1].split(".")[0])
                    except ValueError:
                        continue

                    imgs.append(torch.from_numpy(crop))
                    true_labels.append(true_label)

                if len(imgs) == 0:
                    continue

                preds = self.predict(imgs)

                y_true.extend(true_labels)
                y_pred.extend(preds.tolist())

        self.cm_counts = confusion_matrix(y_true, y_pred)
        self.cm_norms = confusion_matrix(y_true, y_pred, normalize='true')
        self.accuracy = (self.cm_counts.diagonal().sum() / self.cm_counts.sum()) * 100
        

    
    def get_cm_counts(self):
        return self.cm_counts

    def get_confusion_matrix(self):
        return self.cm_norms 
    
    def get_accuracy(self):
        return self.accuracy
    
    def draw_confusion_matrix(self):
        """
        Génère une visualisation graphique de la matrice de confusion.
        
        Affiche les pourcentages de réussite et les erreurs de prédiction 
        sous forme de carte de chaleur (heatmap) avec Matplotlib.
        """
        if self.cm_norms is None:
            raise ValueError("Confusion matrix not computed yet. Run fast_identify first.")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(self.cm_norms, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (Accuracy: {self.accuracy:.2f}%)')
        plt.colorbar()
        tick_marks = np.arange(len(self.cm_norms))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)

        thresh = self.cm_norms.max() / 2.
        for i in range(self.cm_norms.shape[0]):
            for j in range(self.cm_norms.shape[1]):
                plt.text(j, i, f"{self.cm_counts[i, j]}",
                         horizontalalignment="center",
                         color="white" if self.cm_norms[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
