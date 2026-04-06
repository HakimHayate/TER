import numpy as np
from pdf2image import convert_from_path
import os

def convertir_pdf_en_images(pdf_path, dpi=300):
    coeurs = os.cpu_count() or 4
    
    try:
        # On ne précise plus poppler_path ! Le système va le chercher tout seul.
        pages_pil = convert_from_path(pdf_path, dpi=dpi, thread_count=coeurs)
    except Exception as e:
        print("❌ ERREUR : Poppler n'est pas trouvé. Vérifiez qu'il est installé et ajouté au PATH du système.")
        raise e
        
    pages_cv = []
    for page in pages_pil:
        open_cv_image = np.array(page)
        image_bgr = open_cv_image[:, :, ::-1].copy()
        pages_cv.append(image_bgr)
        
    return pages_cv