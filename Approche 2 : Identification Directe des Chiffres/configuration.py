# --- CHEMINS DES FICHIERS (INPUT/OUTPUT) ---
# Le PDF source à traiter
pdf_path = "input/1e-r-0.pdf"

# Le fichier JSON contenant les coordonnées théoriques des cases (en mm)
json_path = "data/atomic-boxes.json"

# Dossier où seront enregistrées les cases découpées pour vérification
identify_path = "output/crop"

# Dossier racine pour accumuler les images destinées au futur fine-tuning
finetuning_data_path = "finetuning_data"


# --- MODÈLES IA ---
# Le checkpoint PyTorch (utilisé par finetuning.py pour charger les poids à 88%)
model_path_pth = "model.pth"

# Le modèle final optimisé (utilisé par identification.py pour l'inférence rapide)
model_path = "model_static_finetune.onnx"


# --- PARAMÈTRES GÉOMÉTRIQUES & SCANNER ---
# Dimensions standards d'une feuille A4 en millimètres
img_width = 210.0
img_height = 297.0

# Facteur de redimensionnement pour l'affichage ou le calcul (1 = taille réelle)
scale = 1


# --- PARAMÈTRES DE VISION (OPENCV) ---
# Seuil de binarisation pour la détection des QR Codes et des repères.
# 237 est une valeur haute, idéale pour isoler le noir sur un papier très blanc.
threshold = 237