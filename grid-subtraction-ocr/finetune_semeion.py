import os
import urllib.request
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# ==========================================
# CONFIGURATION
# ==========================================
URL_SEMEION = "https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
FICHIER_LOCAL = "semeion.data"
MODELE_BASE = "modele_emnist.h5"
NOUVEAU_MODELE = "modele_europe.h5"

def telecharger_semeion():
    """Télécharge le dataset Semeion si on ne l'a pas déjà."""
    if not os.path.exists(FICHIER_LOCAL):
        print("🌐 Téléchargement du dataset Semeion (Européen) depuis l'Université d'Irvine...")
        urllib.request.urlretrieve(URL_SEMEION, FICHIER_LOCAL)
        print("✅ Téléchargement terminé !")
    else:
        print("✅ Fichier Semeion déjà présent en local.")

def preparer_donnees_semeion():
    """Lit le fichier texte Semeion et le transforme en images 28x28 pour Keras."""
    print("🔄 Conversion des données Semeion (16x16) vers le format IA (28x28)...")
    
    images = []
    labels = []
    
    with open(FICHIER_LOCAL, "r") as f:
        lignes = f.readlines()
        
    for ligne in lignes:
        valeurs = ligne.strip().split()
        if len(valeurs) != 266:
            continue
            
        # Les 256 premières valeurs sont les pixels (grille 16x16)
        pixels = np.array(valeurs[:256], dtype=np.float32).reshape(16, 16)
        
        # Les 10 dernières valeurs sont les labels (format "One-Hot", ex: 0 0 1 0 0... = chiffre 2)
        one_hot_label = np.array(valeurs[256:], dtype=np.float32)
        vrai_label = np.argmax(one_hot_label)
        
        # --- L'ASTUCE DE REDIMENSIONNEMENT ---
        # Semeion est en 16x16. On le grossit en 20x20, puis on le centre sur un fond noir 28x28
        # Cela imite parfaitement la façon dont EMNIST a été créé !
        pixels_agrandis = cv2.resize(pixels, (20, 20), interpolation=cv2.INTER_CUBIC)
        
        fond_noir = np.zeros((28, 28), dtype=np.float32)
        # On colle l'image 20x20 au centre du 28x28 (marge de 4 pixels de chaque côté)
        fond_noir[4:24, 4:24] = pixels_agrandis
        
        images.append(fond_noir)
        labels.append(vrai_label)
        
    X = np.array(images).reshape(-1, 28, 28, 1)
    y = np.array(labels)
    
    print(f"✅ {len(X)} images européennes prêtes pour l'entraînement.")
    return X, y

def main():
    print("=== 🧠 DÉMARRAGE DU FINE-TUNING EUROPÉEN ===")
    
    telecharger_semeion()
    X, y = preparer_donnees_semeion()
    
    # On garde 20% des Italiens pour vérifier que l'IA ne fait pas juste du "par cœur"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if not os.path.exists(MODELE_BASE):
        print(f"❌ Erreur : Modèle de base '{MODELE_BASE}' introuvable.")
        return
        
    print(f"\n🤖 Chargement de ton modèle actuel : {MODELE_BASE}")
    modele = load_model(MODELE_BASE)
    
    # --- LA SÉCURITÉ DU FINE-TUNING ---
    # On recompile avec un "Learning Rate" minuscule (0.0001). 
    # Si on le laisse normal, l'IA va oublier tout EMNIST et ne connaître QUE Semeion.
    modele.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🚀 Apprentissage de l'écriture européenne en cours...")
    historique = modele.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,          # 15 passages sur les 1593 images
        batch_size=32,
        verbose=1
    )
    
    modele.save(NOUVEAU_MODELE)
    print("\n" + "="*50)
    print(f"🎉 OPÉRATION RÉUSSIE ! Modèle sauvegardé sous : {NOUVEAU_MODELE}")
    print("="*50)

if __name__ == "__main__":
    main()