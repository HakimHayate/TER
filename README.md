# Hekzam-FormDetection
A computer vision project for detecting and extracting form fields from automatically scanned exam sheets affected by non-linear distortions, using OpenCV
Python 3.12.3

**⚙️ Utilisation (Production)**
Pour lancer l'identification automatique sur un PDF en utilisant le multiprocessing :
    python main.py
    
Le script va :

Découper le PDF en segments.
Lancer les Producteurs pour le rendu et le découpage des cases.
Alimenter le Consommateur pour l'identification via ONNX Runtime.

**📂 Structure du Projet**

    identification/ : Moteur d'inférence ONNX et métriques (Matrice de confusion).
    preprocess/ : Algorithmes d'homographie, découpage et normalisation MNIST.
    model/ : Architecture CNN, scripts de Train et de Fine-tuning.
    configuration/ : Paramètres globaux et chemins des templates JSON.
    utils/ : Helpers pour la gestion des PDF et des fichiers.