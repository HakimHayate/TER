import os
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import concurrent.futures 
from tensorflow.keras.models import load_model

def extraire_vrai_label(nom_fichier):
    """Extrait le label depuis le nom du fichier (ex: id-0-1-9.png -> 9)"""
    try:
        base = os.path.splitext(nom_fichier)[0]
        parties = base.split('-')
        dernier = parties[-1]
        if dernier.isdigit():
            return int(dernier)
    except:
        pass
    return None


def charger_et_preparer_image(chemin_img):
    """Fonction ultra-rapide pour lire et préparer une image (utilisée par le Multithreading)"""
    nom_fichier = os.path.basename(chemin_img)
    vrai_label = extraire_vrai_label(nom_fichier)
    
    if vrai_label is None:
        return None
        
    img = cv2.imread(chemin_img, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        return None
        
    # Redimensionnement et normalisation
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    
    return (img, vrai_label, chemin_img)


def evaluer_modele(dossier_sortie, chemin_modele="modele_emnist.h5"):
    print("\n" + "="*50)
    print("🤖 CHARGEMENT DU MODÈLE ET ÉVALUATION RAPIDE")
    print("="*50)
    
    if not os.path.exists(chemin_modele):
        print(f"❌ Erreur : Modèle introuvable au chemin '{chemin_modele}'")
        return

    modele = load_model(chemin_modele)
    fichiers_images = glob.glob(os.path.join(dossier_sortie, "**", "*.png"), recursive=True)
    
    if not fichiers_images:
        print("⚠️ Aucune image à évaluer trouvée dans le dossier de sortie.")
        return

    print(f"📂 {len(fichiers_images)} fichiers trouvés. Lecture depuis le disque...")
    
    # 1. LECTURE MULTITHREAD (Très rapide)
    images_pretes = []
    y_vrai = []
    chemins_valides = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        resultats = list(tqdm(executor.map(charger_et_preparer_image, fichiers_images), 
                              total=len(fichiers_images), 
                              desc="Préparation des images", 
                              unit="img"))

    # Filtrage des images valides
    for res in resultats:
        if res is not None:
            images_pretes.append(res[0])
            y_vrai.append(res[1])
            chemins_valides.append(res[2])

    if not images_pretes:
        print("⚠️ Aucune donnée évaluable (Vérifie le nom de tes fichiers PNG).")
        return

    # 2. PRÉDICTION EN BATCH (Le secret de la vitesse)
    print("\n⚡ Reconnaissance IA en cours (Batch processing)...")
    X = np.array(images_pretes).reshape(-1, 28, 28, 1) # Format pour Keras
    y_vrai = np.array(y_vrai)
    
    
    # On donne tout le paquet à l'IA d'un seul coup
    predictions = modele.predict(X, batch_size=256)
    y_pred = np.argmax(predictions, axis=1)


    # 3. CALCUL DES STATISTIQUES
    total = len(y_vrai)
    corrects = np.sum(y_vrai == y_pred)
    precision = (corrects / total) * 100
    
    print("\n" + "="*50)
    print(f"🎯 SCORE FINAL : {corrects}/{total} ({precision:.2f}%)")
    print("="*50)
    
    # Recherche d'erreurs pour l'affichage
    erreurs = [(chemins_valides[i], y_vrai[i], y_pred[i]) for i in range(total) if y_vrai[i] != y_pred[i]]
    
    if erreurs:
        print("\n🧐 Exemples d'erreurs (5 au hasard) :")
        exemples = random.sample(erreurs, min(5, len(erreurs)))
        for chemin, vrai, predit in exemples:
            nom_fichier = os.path.basename(chemin)
            dossier_parent = os.path.basename(os.path.dirname(os.path.dirname(chemin)))
            print(f"   ❌ {dossier_parent}/{nom_fichier} | Vrai : {vrai} | L'IA a lu : {predit}")

    print("\n[DÉTAIL PAR CHIFFRE]")
    print(classification_report(y_vrai, y_pred))

    # 4. MATRICE DE CONFUSION
    #generer_matrice_confusion(y_vrai, y_pred, precision)


def generer_matrice_confusion(y_vrai, y_pred, precision):
    """Crée et sauvegarde la matrice de confusion en PNG."""
    print("\n📊 Génération de la matrice de confusion...")
    
    labels_possibles = list(range(10))
    cm = confusion_matrix(y_vrai, y_pred, labels=labels_possibles)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_possibles, yticklabels=labels_possibles)
    
    plt.xlabel('Prédiction de l\'IA (Ce que l\'ordinateur a lu)', fontsize=12)
    plt.ylabel('Vérité Terrain (Ce qui était vraiment écrit)', fontsize=12)
    plt.title(f'Matrice de Confusion OCR (Précision globale : {precision:.1f}%)', fontsize=14)
    
    chemin_sauvegarde = "matrice_confusion.png"
    plt.savefig(chemin_sauvegarde, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✅ Matrice sauvegardée avec succès sous : '{chemin_sauvegarde}'")