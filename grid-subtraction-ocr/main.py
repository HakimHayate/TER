import os
import glob
import json
import argparse
from tqdm import tqdm  
from src.pdf_utils import convertir_pdf_en_images
from src.vision_utils import traiter_page_et_decouper
from src.inference_utils import evaluer_modele
import concurrent.futures  

DOSSIER_SCANS = "data/scans"
DOSSIER_SORTIE = "data/extractions"
JSON_PATH = "data/atomic-boxes.json"

def etape_extraction():
    print("\n" + "="*40)
    print("🚀 ÉTAPE 1 : EXTRACTION & PRÉPARATION MNIST")
    print("="*40)
    
    if not os.path.exists(JSON_PATH):
        print(f"❌ Erreur : Le fichier JSON '{JSON_PATH}' est introuvable.")
        return
        
    with open(JSON_PATH, 'r') as f:
        data_json = json.load(f)

    fichiers_pdf = glob.glob(os.path.join(DOSSIER_SCANS, "*.pdf"))
    if not fichiers_pdf:
        print(f"⚠️ Aucun PDF trouvé dans '{DOSSIER_SCANS}'.")
        return

    barre_pdfs = tqdm(fichiers_pdf, desc="Documents traités", unit="pdf")
    
    for pdf_path in barre_pdfs:
        nom_pdf = os.path.basename(pdf_path).replace(".pdf", "")
        barre_pdfs.set_postfix({"Fichier en cours": nom_pdf})
        
        try:
            pages_cv = convertir_pdf_en_images(pdf_path, dpi=300)
            def traiter_une_page(args):
                idx, page_img = args
                numero_page = idx + 1
                dossier_sauvegarde = os.path.join(DOSSIER_SORTIE, nom_pdf, f"Page_{numero_page}")
                traiter_page_et_decouper(page_img, numero_page, data_json, dossier_sauvegarde)

            taches = list(enumerate(pages_cv))

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(tqdm(executor.map(traiter_une_page, taches), 
                          total=len(taches), 
                          desc=f"Pages de {nom_pdf}", 
                          leave=False, 
                          unit="page"))

        except Exception as e:
            tqdm.write(f"❌ Erreur sur {nom_pdf} : {str(e)}")

def etape_reconnaissance():
    print("\n" + "="*40)
    print("🤖 ÉTAPE 2 : RECONNAISSANCE IA")
    print("="*40)
    
    if not os.path.exists(DOSSIER_SORTIE) or not os.listdir(DOSSIER_SORTIE):
        print("⚠️ Le dossier d'extraction est vide. Lance d'abord l'extraction !")
        return
        
    evaluer_modele(DOSSIER_SORTIE, chemin_modele="modele_emnist.h5")

def main():
    parser = argparse.ArgumentParser(description="Pipeline OCR")
    parser.add_argument("--mode", type=str, choices=["extract", "recognize", "all"], default="all")
    args = parser.parse_args()

    if args.mode in ["extract", "all"]:
        etape_extraction()

    if args.mode in ["recognize", "all"]:
        etape_reconnaissance()

if __name__ == "__main__":
    main()