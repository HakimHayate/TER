import numpy as np
import json
import cv2
from pyzbar.pyzbar import decode  
import configuration as cf
import time  
import os
import fitz



def stream_pdf_pages(pdf_path, start_page=0, end_page=None):
    """
    Générateur extrayant les flux d'images brutes d'une plage de pages PDF.
    
    Optimise la mémoire en ne chargeant pas le document complet et en 
    décodant les images à la volée via OpenCV pour le traitement.

    Args:
        pdf_path (str): Chemin local vers le fichier PDF à traiter.
        start_page (int): Index de la première page (inclus, défaut 0).
        end_page (int, optional): Index de fin (exclu). Si None, traite jusqu'à la fin.

    Yields:
        numpy.ndarray: Image décodée au format BGR (OpenCV) prête pour le prétraitement.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    if end_page is None or end_page > total_pages:
        end_page = total_pages

    for page_index in range(start_page, end_page):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                yield img 
    doc.close()


def read_json(json_path):
    """
    Extrait les coordonnées théoriques (en mm) des marqueurs de calage et des zones de saisie.

    Cette fonction parse le fichier de configuration JSON pour définir les points d'ancrage 
    nécessaires à l'homographie et la liste complète des rectangles à découper.

    Args:
        json_path (str): Chemin vers le fichier JSON contenant le template du formulaire.

    Returns:
        tuple: Un tuple contenant :
            - list: Les 8 points d'ancrage (Top-Left, Top-Right, etc.) pour la matrice de transformation.
            - list: La liste de tous les rectangles au format (x, y, w, h).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        tltl = (data["marker barcode tl page1"]["x"], data["marker barcode tl page1"]["y"])
        tlbr = (data["marker barcode tl page1"]["x"] + data["marker barcode tl page1"]["width"], data["marker barcode tl page1"]["y"] + data["marker barcode tl page1"]["height"])
        trtl = (data["marker barcode tr page1"]["x"],data["marker barcode tr page1"]["y"])
        trbr = (data["marker barcode tr page1"]["x"] + data["marker barcode tr page1"]["width"],data["marker barcode tr page1"]["y"] + data["marker barcode tr page1"]["height"])
        bltl = (data["marker barcode bl page1"]["x"],data["marker barcode bl page1"]["y"])
        blbr = (data["marker barcode bl page1"]["x"] + data["marker barcode bl page1"]["width"], data["marker barcode bl page1"]["y"] + data["marker barcode bl page1"]["height"])
        brtl = (data["marker barcode br page1"]["x"], data["marker barcode br page1"]["y"])
        brbr = (data["marker barcode br page1"]["x"] + data["marker barcode br page1"]["width"],data["marker barcode br page1"]["y"] + data["marker barcode br page1"]["height"])


        rects = []
        for key, value in data.items():
            x = value.get("x", 0)
            y = value.get("y", 0)
            w = value.get("width", 0)
            h = value.get("height", 0)
            rects.append((x, y, w, h))
        return [tltl,tlbr,trtl,trbr,bltl,blbr,brtl,brbr], rects

def save_boxes(pages_dir, crop_img,box_idx):
    """
    Enregistre physiquement une image découpée (crop) sur le disque.

    Utilise l'identifiant extrait du JSON pour nommer le fichier, en 
    tronquant les identifiants longs pour garantir la compatibilité du nommage.

    Args:
        pages_dir (str): Répertoire de destination (généralement lié à l'index de la page).
        boxe (numpy.ndarray): Matrice de pixels de la zone découpée.
        box_idx (str): Identifiant unique de la case (ex: clé JSON) servant de nom de fichier.

    Returns:
        None: Le fichier est écrit directement dans le dossier spécifié.
    """
    if len(box_idx) < 10 :
        box_idx = box_idx[-5:]

    crop_path = os.path.join(pages_dir,f"{box_idx}.jpg")
    cv2.imwrite(crop_path, crop_img)




def read_Qrcode(img):
    """
    Localise et décode les 4 QR codes de calage situés aux coins du document.
    
    Divise l'image en zones d'intérêt (ROI) de 20% pour accélérer la détection 
    et extrait les coordonnées (x, y) de chaque marqueur pour recaler le formulaire.

    Args:
        img (numpy.ndarray): L'image de la page entière en format BGR ou Gris.

    Returns:
        list: Liste de 8 couples de coordonnées (pixels) correspondant aux coins 
              extérieurs et intérieurs des QR codes détectés. Les emplacements non 
              détectés restent à None.
    """
    detector = cv2.QRCodeDetector()
    result = [False, False, False, False]
    src = [None]*8

    h, w = img.shape[:2]

    # ---------- TOP LEFT ----------
    x_offset = 0
    y_offset = 0

    img_tl = img[:int(0.2*h), :int(0.2*w)]
    data, points, _ = detector.detectAndDecode(img_tl)

    if points is not None:
        result[0] = True
        pts = points[0]
        x1, y1 = pts[0]
        x2, y2 = pts[2]

        src[0] = (int(x1 + x_offset), int(y1 + y_offset))
        src[1] = (int(x2 + x_offset), int(y2 + y_offset))


    # ---------- TOP RIGHT ----------
    x_offset = int(0.8*w)
    y_offset = 0

    img_tr = img[:int(0.2*h), int(0.8*w):]
    data, points, _ = detector.detectAndDecode(img_tr)

    if points is not None:
        result[1] = True
        pts = points[0]
        x1, y1 = pts[0]
        x2, y2 = pts[2]

        src[2] = (int(x1 + x_offset), int(y1 + y_offset))
        src[3] = (int(x2 + x_offset), int(y2 + y_offset))


    # ---------- BOTTOM LEFT ----------
    x_offset = 0
    y_offset = int(0.8*h)

    img_bl = img[int(0.8*h):, :int(0.2*w)]
    data, points, _ = detector.detectAndDecode(img_bl)

    if points is not None:
        result[2] = True
        pts = points[0]
        x1, y1 = pts[0]
        x2, y2 = pts[2]

        src[4] = (int(x1 + x_offset), int(y1 + y_offset))
        src[5] = (int(x2 + x_offset), int(y2 + y_offset))


    # ---------- BOTTOM RIGHT ----------
    x_offset = int(0.8*w)
    y_offset = int(0.8*h)

    img_br = img[int(0.8*h):, int(0.8*w):]
    data, points, _ = detector.detectAndDecode(img_br)

    if points is not None:
        result[3] = True
        pts = points[0]
        x1, y1 = pts[0]
        x2, y2 = pts[2]

        src[6] = (int(x1 + x_offset), int(y1 + y_offset))
        src[7] = (int(x2 + x_offset), int(y2 + y_offset))


    return src



def transform_rects(rects, H):
    """
    Projette les rectangles théoriques sur l'image scannée via une matrice d'homographie.
    
    Convertit chaque zone (x, y, w, h) en 4 points, applique la transformation de 
    perspective, puis recalcule la boîte englobante droite (Bounding Box) 
    alignée sur les axes de l'image de destination.

    Args:
        rects (list): Liste de tuples (x, y, w, h) en pixels (coordonnées théoriques).
        H (numpy.ndarray): Matrice d'homographie 3x3 calculée via cv2.findHomography.

    Returns:
        list: Liste de tuples (xmin, ymin, xmax, ymax) représentant les coordonnées 
              projetées et redressées prêtes pour l'opération de découpage (crop).
    """
    transformed_rects = []

    for rect in rects:
        x, y, w, h = rect
        corners = np.array([[x, y],
                            [x+w, y],
                            [x+w, y+h],
                            [x, y+h]], dtype=np.float32)

        corners = corners.reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)

        transformed_corners = transformed_corners.reshape(-1, 2)
        xmin, ymin = np.min(transformed_corners, axis=0)
        xmax, ymax = np.max(transformed_corners, axis=0)
        

        transformed_rects.append((int(xmin), int(ymin), int(xmax), int(ymax)))

    return transformed_rects

def mm_to_pixel_list(coords_mm,img):
    """
    Convertit des coordonnées du monde réel (mm) en coordonnées image (pixels).
    
    Calcule dynamiquement les ratios d'échelle en fonction de la résolution 
    de l'image fournie et des dimensions de référence définies dans la configuration.

    Args:
        coords_mm (list): Liste de tuples (x, y) ou (x, y, w, h) exprimés en millimètres.
        img (numpy.ndarray): Image de référence utilisée pour extraire la résolution (H, W).

    Returns:
        list: Liste de coordonnées converties en entiers (pixels), arrondie à l'unité 
              la plus proche pour garantir la précision du découpage.

    Raises:
        ValueError: Si une coordonnée ne possède ni 2 ni 4 éléments.
    """
    img_height_px, img_width_px = img.shape[:2]
    scale_x = img_width_px / cf.img_width
    scale_y = img_height_px / cf.img_height

    coords_px = []
    for c in coords_mm:
        if len(c) == 2:  
            x_px = int(round(c[0] * scale_x))
            y_px = int(round(c[1] * scale_y))
            coords_px.append((x_px, y_px))
        elif len(c) == 4: 
            x_px = int(round(c[0] * scale_x))
            y_px = int(round(c[1] * scale_y))
            w_px = int(round(c[2] * scale_x))
            h_px = int(round(c[3] * scale_y))
            coords_px.append((x_px, y_px, w_px, h_px))
        else:
            raise ValueError("Chaque coordonnée doit être (x,y) ou (x,y,w,h)")
    return coords_px



def filter_valid_markers(src, dst):
    """
    Filtre et aligne les points d'ancrage théoriques et détectés.
    
    Élimine les paires de coordonnées incomplètes (cas où un marqueur 
    n'a pas été détecté sur l'image) pour garantir la validité de 
    la matrice de transformation.

    Args:
        src (list): Liste des points théoriques (source) en pixels.
        dst (list): Liste des points réellement détectés (destination) sur le scan.

    Returns:
        tuple: Un tuple (new_src, new_dst) ne contenant que les paires de points 
              valides (non-None) prêtes pour le calcul de l'homographie.
    """
    new_src = []
    new_dst = []

    for s, d in zip(src, dst):
        if d is not None:
            new_src.append(s)
            new_dst.append(d)

    return new_src, new_dst




def export_processed_images(pages, base_folder="output", sub_folder="preprocess"):
    """
    Exporte une liste d'images traitées vers un répertoire local.
    
    Gère la création automatique de l'arborescence des dossiers et convertit 
    les images du format de travail (RGB) vers le format de stockage (BGR) 
    requis par OpenCV pour une restitution correcte des couleurs.

    Args:
        pages (list): Liste de matrices d'images (numpy.ndarray) à sauvegarder.
        base_folder (str): Dossier racine pour l'export (défaut "output").
        sub_folder (str): Sous-répertoire spécifique (défaut "preprocess").

    Returns:
        None: Affiche un message de confirmation dans la console après l'écriture.
    """
    output_folder = os.path.join(base_folder, sub_folder)
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(pages):
        filename = os.path.join(output_folder, f"{i}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"{len(pages)} images enregistrées dans {output_folder}")
