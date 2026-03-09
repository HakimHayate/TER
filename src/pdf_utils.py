import numpy as np
from pdf2image import convert_from_path
import os
def convertir_pdf_en_images(pdf_path, dpi=300):
    # On détermine le nombre de cœurs de ton processeur
    coeurs = os.cpu_count() or 4
    
    # --- NOUVEAUTÉ : thread_count ---
    # Poppler va maintenant utiliser tous tes cœurs pour convertir le PDF
    pages_pil = convert_from_path(pdf_path, dpi=dpi, thread_count=coeurs, poppler_path=r"C:\poppler-25.12.0\Library\bin")
    
    pages_cv = []
    for page in pages_pil:
        open_cv_image = np.array(page)
        image_bgr = open_cv_image[:, :, ::-1].copy()
        pages_cv.append(image_bgr)
        
    return pages_cv