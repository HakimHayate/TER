import os
import cv2
import json
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model

# Nouveaux imports pour la matrice
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import de TES fonctions
from src.vision_utils import traiter_page_et_decouper
from src.pdf_utils import convertir_pdf_en_images  

CHEMIN_PDF_TEST = "data/scans/1e-r-1.pdf" 
CHEMIN_JSON = "data/atomic-boxes.json"
CHEMIN_MODELE = "modele_europe.h5"
DOSSIER_TEMP = "data/temp_grid_search"

grille_parametres = {
    'thresh_block': [35],      
    'thresh_c': [1],             
    'hough_thresh': [40],          
    'hough_min_len': [80],         
    'kernel_morph': [54]           
}

def extraire_vrai_label(nom_fichier):
    try:
        base = os.path.splitext(nom_fichier)[0]
        parties = base.split('-')
        dernier = parties[-1]
        if dernier.isdigit(): return int(dernier)
    except: pass
    return None

def evaluer_dossier_temp(dossier, modele):
    """Évalue le dossier et retourne l'accuracy, les vrais labels et les prédictions."""
    fichiers = [os.path.join(r, f) for r, d, fs in os.walk(dossier) for f in fs if f.endswith(".png")]
    if not fichiers: return 0.0, [], []

    y_vrai, images = [], []
    for chemin in fichiers:
        vrai_label = extraire_vrai_label(os.path.basename(chemin))
        if vrai_label is None: continue
        img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        images.append(img)
        y_vrai.append(vrai_label)

    if not images: return 0.0, [], []

    X = np.array(images).reshape(-1, 28, 28, 1)
    y_vrai = np.array(y_vrai)
    predictions = modele.predict(X, batch_size=256, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    corrects = np.sum(y_vrai == y_pred)
    accuracy = (corrects / len(y_vrai)) * 100
    
    return accuracy, y_vrai, y_pred

def main():
    print("=== 🚀 DÉMARRAGE DE L'OPTIMISATION DES PARAMÈTRES ===")
    
    if not os.path.exists(CHEMIN_PDF_TEST):
        print(f"❌ PDF de test introuvable : {CHEMIN_PDF_TEST}")
        return
        
    print(f"📄 Conversion du PDF '{os.path.basename(CHEMIN_PDF_TEST)}' en image...")
    pages_cv = convertir_pdf_en_images(CHEMIN_PDF_TEST, dpi=300)
    
    if not pages_cv:
        print("❌ Erreur lors de la conversion du PDF.")
        return
        
    image_test = pages_cv[0]
    print("✅ Page 1 extraite avec succès pour le test.")

    with open(CHEMIN_JSON, 'r') as f:
        data_json = json.load(f)
        
    print("🤖 Chargement du modèle IA...")
    modele = load_model(CHEMIN_MODELE)

    cles = grille_parametres.keys()
    valeurs = grille_parametres.values()
    combinaisons = [dict(zip(cles, v)) for v in itertools.product(*valeurs)]
    
    print(f"📊 Nombre total de configurations à tester : {len(combinaisons)}")
    resultats = []

    for i, params in enumerate(tqdm(combinaisons, desc="Tests en cours")):
        if os.path.exists(DOSSIER_TEMP):
            shutil.rmtree(DOSSIER_TEMP)
        os.makedirs(DOSSIER_TEMP)
        
        traiter_page_et_decouper(
            image=image_test.copy(), 
            numero_page=1, 
            data_json=data_json, 
            dossier_sauvegarde=DOSSIER_TEMP, 
            params=params
        )
        
        accuracy, y_vrai, y_pred = evaluer_dossier_temp(DOSSIER_TEMP, modele)
        
        resultats.append({
            'accuracy': accuracy,
            'params': params,
            'y_vrai': y_vrai,
            'y_pred': y_pred
        })

    resultats_tries = sorted(resultats, key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "="*50)
    print("🏆 TOP 3 DES MEILLEURES CONFIGURATIONS")
    print("="*50)
    
    for i in range(min(3, len(resultats_tries))):
        res = resultats_tries[i]
        print(f"\n🥇 Rang {i+1} : SCORE = {res['accuracy']:.2f}%")
        print("Paramètres :")
        for k, v in res['params'].items():
            print(f"   - {k} : {v}")


    meilleur_resultat = resultats_tries[0]
    if len(meilleur_resultat['y_vrai']) > 0:
        print("\n📊 Génération de la matrice de confusion pour la meilleure configuration...")
        
        labels_possibles = list(range(10))
        cm = confusion_matrix(meilleur_resultat['y_vrai'], meilleur_resultat['y_pred'], labels=labels_possibles)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels_possibles, yticklabels=labels_possibles)
        
        plt.xlabel('Prédiction IA', fontsize=12)
        plt.ylabel('Vérité Terrain', fontsize=12)
        plt.title(f"Matrice de Confusion (Meilleur Score : {meilleur_resultat['accuracy']:.1f}%)", fontsize=14)
        
        chemin_matrice = "meilleure_matrice_confusion.png"
        plt.savefig(chemin_matrice, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Matrice sauvegardée avec succès sous : '{chemin_matrice}'")
    else:
        print("\n⚠️ Impossible de générer la matrice (aucun chiffre trouvé).")

    # Nettoyage final
    if os.path.exists(DOSSIER_TEMP):
        shutil.rmtree(DOSSIER_TEMP)

if __name__ == "__main__":
    main()