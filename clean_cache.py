import os
import shutil

# Chemins classiques où EMNIST cache ses fichiers corrompus sur Windows
chemins_possibles = [
    os.path.expanduser("~/.cache/emnist"),
    os.path.join(os.getenv('LOCALAPPDATA', ''), 'emnist')
]

for chemin in chemins_possibles:
    if os.path.exists(chemin):
        shutil.rmtree(chemin)
        print(f"✅ Dossier corrompu supprimé : {chemin}")
        
print("Tu peux maintenant relancer ton entraînement !")