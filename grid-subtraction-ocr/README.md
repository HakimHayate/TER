# 📄 OCR-Auto-Grader : Numérisation Automatisée de Notes d'Examen

Ce projet a été développé dans le cadre d'un projet de Master (TER). Il propose un pipeline hybride complet pour l'extraction et la reconnaissance de notes manuscrites à partir de copies d'examens numérisées (PDF). 

L'approche combine la rigueur géométrique de la **Vision par Ordinateur** (OpenCV) avec la capacité de généralisation de l'**Apprentissage Profond** (CNN/TensorFlow), tout en garantissant des temps de traitement optimisés grâce à une architecture *In-Memory*.

## ✨ Fonctionnalités et Architecture du Pipeline

Notre méthode se démarque par sa robustesse face aux bruits de numérisation et aux chevauchements d'encre :

1. **Alignement par Ancrage Fiduciaire :** Détection de 4 QR Codes aux coins de la page et application d'une matrice d'homographie pour un redressement au pixel près, corrigeant les déformations du scanner.
2. **Nettoyage Structurel Morphologique :** Séparation directionnelle et utilisation de la Transformée de Hough pour isoler puis soustraire la grille du tableau de notation, laissant l'encre manuscrite intacte.
3. **Extraction Déterministe In-Memory :** Découpage des cases basé sur un mapping JSON (`atomic-boxes.json`) et transfert direct des matrices NumPy vers le modèle d'IA (zéro écriture disque intermédiaire pour la prédiction).
4. **Classification IA et Heuristique :** Prédiction par un modèle CNN pré-entraîné sur EMNIST (Batch Processing) et application d'un filtre géométrique post-prédiction (analyse de l'Aspect Ratio) pour corriger les biais culturels liés à la graphie européenne (ex: confusion 1/4).

---

## 🛠️ Prérequis et Installation

### 1. Environnement Python
Assurez-vous d'avoir **Python 3.8 ou supérieur** installé. Il est recommandé d'utiliser un environnement virtuel (`venv` ou `conda`).

Clonez le dépôt et installez les dépendances :
```bash
git clone https://github.com/HakimHayate/TER
cd grid-subtraction-ocr
pip install -r requirements.txt
python main.py --mode all
