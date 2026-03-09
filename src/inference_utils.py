import os
import glob
import cv2
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures # <-- Le secret pour lire plusieurs fichiers en même temps

def preparer_image_pour_mnist(chemin_image):
    # (Garde la fonction intacte si tu t'en sers encore ailleurs)
    pass 

# NOUVELLE FONCTION POUR LE THREADING
def lire_une_image(chemin):
    """Fonction qui lit une seule image rapidement depuis le disque"""
    label_vrai_str = os.path.basename(os.path.dirname(chemin))
    if not label_vrai_str.isdigit():
        return None
        
    img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    return (img, int(label_vrai_str), chemin)


def evaluer_modele(dossier_extractions, chemin_modele="modele_custom.h5"):
    if not os.path.exists(chemin_modele):
        print(f"\n❌ Erreur : Le modèle '{chemin_modele}' est introuvable.")
        return

    print(f"\n🧠 Chargement du réseau de neurones ({chemin_modele})...")
    model = tf.keras.models.load_model(chemin_modele)

    chemins_images = glob.glob(os.path.join(dossier_extractions, "*", "*", "*", "*.png"))
    
    if not chemins_images:
        print("⚠️ Aucune image trouvée pour l'évaluation.")
        return

    print("\n📂 Chargement ULTRA-RAPIDE des images en mémoire (Multithreading)...")
    
    images_brutes = []
    labels_vrais = []
    chemins_valides = []

    # 1. LECTURE PARALLÈLE (Multithreading)
    # L'ordinateur va utiliser tous ses cœurs pour lire le disque dur
    with concurrent.futures.ThreadPoolExecutor() as executor:
        resultats = list(tqdm(executor.map(lire_une_image, chemins_images), 
                              total=len(chemins_images), 
                              desc="Lecture disque", 
                              unit="img"))

    # 2. FILTRAGE
    for res in resultats:
        if res is not None:
            images_brutes.append(res[0])
            labels_vrais.append(res[1])
            chemins_valides.append(res[2])

    if not images_brutes:
        print("⚠️ Aucune image valide n'a pu être chargée.")
        return

    print("⚡ Vectorisation et Normalisation...")
    # 3. VECTORISATION NUMPY (Calcul matriciel massif)
    # Au lieu de faire la division par 255 dans une boucle pour chaque image, 
    # on le fait en UNE SEULE OPÉRATION sur l'énorme bloc de données !
    X_test = np.array(images_brutes, dtype="float32")
    X_test = X_test / 255.0
    X_test = np.expand_dims(X_test, axis=-1)
    
    y_test = np.array(labels_vrais)

    print("\n🚀 Prédiction IA en cours (Batch processing)...")
    predictions = model.predict(X_test, batch_size=256)
    labels_predits = np.argmax(predictions, axis=1)

    # --- CALCUL DES RÉSULTATS ---
    predictions_correctes = 0
    erreurs = []

    for i in range(len(y_test)):
        if labels_predits[i] == y_test[i]:
            predictions_correctes += 1
        else:
            erreurs.append((chemins_valides[i], y_test[i], labels_predits[i]))

    total_images = len(y_test)
    precision = (predictions_correctes / total_images) * 100
    
    # --- AFFICHAGE ---
    print("\n" + "="*45)
    print("📊 RÉSULTATS DE L'ÉVALUATION OCR")
    print("="*45)
    print(f"Modèle utilisé        : {chemin_modele}")
    print(f"Images analysées      : {total_images:,}")
    print(f"Prédictions correctes : {predictions_correctes:,}")
    print(f"Précision globale     : {precision:.2f} %")
    print("="*45)

    if erreurs:
        print("\n🧐 Exemples d'erreurs (5 au hasard) :")
        exemples = random.sample(erreurs, min(5, len(erreurs)))
        for chemin, vrai, predit in exemples:
            nom_fichier = os.path.basename(chemin)
            dossier_parent = os.path.basename(os.path.dirname(os.path.dirname(chemin)))
            print(f" ❌ {dossier_parent}/{nom_fichier} | Vrai : {vrai} | L'IA a lu : {predit}")