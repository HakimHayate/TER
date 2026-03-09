import cv2
import numpy as np
import os

def traiter_page_et_decouper(image, numero_page, data_json, dossier_sauvegarde):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Binarisation originale (chiffres blancs sur fond noir)
    binary_original = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 35, 1)

    h, w = binary_original.shape
    safe_margin = int(w * 0.09) 

    # ==========================================
    # 2. CRÉATION DU MASQUE DE GRILLE (AVEC HOUGH)
    # ==========================================
    binary_for_grid = binary_original.copy()
    binary_for_grid[0:safe_margin, 0:safe_margin] = 0 
    binary_for_grid[0:safe_margin, w-safe_margin:w] = 0
    binary_for_grid[h-safe_margin:h, w-safe_margin:w] = 0
    binary_for_grid[h-safe_margin:h, 0:safe_margin] = 0

    # A. Peignes morphologiques de base
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (54, 1))
    h_mask_brut = cv2.morphologyEx(binary_for_grid, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 54))
    v_mask_brut = cv2.morphologyEx(binary_for_grid, cv2.MORPH_OPEN, v_kernel)

    h_mask_propre = np.zeros_like(binary_for_grid)
    v_mask_propre = np.zeros_like(binary_for_grid)

    lines_h = cv2.HoughLinesP(h_mask_brut, 1, np.pi/180, threshold=40, minLineLength=80, maxLineGap=80)
    
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            # On dessine un beau trait blanc d'épaisseur 2 sur la toile noire
            cv2.line(h_mask_propre, (x1, y1), (x2, y2), 255, 2)

    lines_v = cv2.HoughLinesP(v_mask_brut, 1, np.pi/180, threshold=40, minLineLength=80, maxLineGap=80)
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            cv2.line(v_mask_propre, (x1, y1), (x2, y2), 255, 2)

    # B. Fusion des lignes parfaites
    fusion_brute = cv2.bitwise_or(h_mask_propre, v_mask_propre) 
    kernel_fuse = np.ones((3,3), np.uint8)
    grid_mask = cv2.dilate(fusion_brute, kernel_fuse, iterations=1)

    # ==========================================
    # 3. Suppression des traits -> Image propre
    # ==========================================
    text_and_qr_inv = cv2.subtract(binary_original, grid_mask)
    final_clean_image = cv2.bitwise_not(text_and_qr_inv)
    # 4. Recherche des QR Codes
    binary_recherche = cv2.bitwise_not(final_clean_image)
    marge_qr = int(w * 0.15)
    
    cles_qr = [
        "marker barcode tl page1",
        "marker barcode tr page1",
        "marker barcode br page1",
        "marker barcode bl page1"
    ]
    
    zones_recherche = {
        cles_qr[0]: (0, 0, marge_qr, marge_qr),
        cles_qr[1]: (w - marge_qr, 0, w, marge_qr),
        cles_qr[2]: (w - marge_qr, h - marge_qr, w, h),
        cles_qr[3]: (0, h - marge_qr, marge_qr, h)
    }

    centres_scan = {}
    kernel_qr = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    for nom, (x1, y1, x2, y2) in zones_recherche.items():
        zone = binary_recherche[y1:y2, x1:x2]
        zone_fermee = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, kernel_qr)
        contours, _ = cv2.findContours(zone_fermee, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            plus_gros_contour = max(contours, key=cv2.contourArea)
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(plus_gros_contour)
            cx = x_rect + (w_rect / 2.0) + x1
            cy = y_rect + (h_rect / 2.0) + y1
            centres_scan[nom] = [cx, cy]

    if len(centres_scan) != 4:
        return # On sort silencieusement car on a la barre de progression

    # 5. Calcul Homographie Inverse
    centres_json = {}
    for cle in cles_qr:
        if cle not in data_json:
            return
        box = data_json[cle]
        centres_json[cle] = [box['x'] + (box['width'] / 2.0), box['y'] + (box['height'] / 2.0)]

    src_points = np.array([centres_scan[k] for k in cles_qr], dtype="float32")
    dst_points = np.array([centres_json[k] for k in cles_qr], dtype="float32")
    H_inverse, _ = cv2.findHomography(dst_points, src_points)

    # 6. Découpage, CENTRAGE MNIST et Sauvegarde
    os.makedirs(dossier_sauvegarde, exist_ok=True)

    for nom_case, infos in data_json.items():
        if "barcode" in nom_case:
            continue 
        
        x, y, w_box, h_box = float(infos['x']), float(infos['y']), float(infos['width']), float(infos['height'])
        
        coins_json = np.array([[[x, y]], [[x + w_box, y]], [[x + w_box, y + h_box]], [[x, y + h_box]]], dtype="float32")
        coins_scan = cv2.perspectiveTransform(coins_json, H_inverse)
        coins_scan = np.int32(coins_scan)
        
        x_b, y_b, w_b, h_b = cv2.boundingRect(coins_scan)
        x_b, y_b = max(0, x_b), max(0, y_b)
        
        chiffre_brut = text_and_qr_inv[y_b:y_b+h_b, x_b:x_b+w_b]
        
        # On trouve la boîte du chiffre pour le recadrer
        contours_chiffre, _ = cv2.findContours(chiffre_brut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_chiffre:
            c = max(contours_chiffre, key=cv2.contourArea)
            xc, yc, wc, hc = cv2.boundingRect(c)
            
            # Si le contour est suffisamment grand (on ignore les micropoussières)
            if wc > 0 and hc > 0:
                chiffre_seul = chiffre_brut[yc:yc+hc, xc:xc+wc]
                kernel_stylo = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                
                chiffre_seul = cv2.dilate(chiffre_seul, kernel_stylo, iterations=1)
                
                chiffre_seul = cv2.morphologyEx(chiffre_seul, cv2.MORPH_CLOSE, kernel_stylo)
                # ----------------------------------------------------

                fond_noir = np.zeros((28, 28), dtype=np.uint8)
                
                facteur = 20.0 / max(wc, hc)
                w_redim = max(1, int(wc * facteur))
                h_redim = max(1, int(hc * facteur))
                
                chiffre_redim = cv2.resize(chiffre_seul, (w_redim, h_redim), interpolation=cv2.INTER_AREA)
                # Centrer
                x_offset = (28 - w_redim) // 2
                y_offset = (28 - h_redim) // 2
                fond_noir[y_offset:y_offset+h_redim, x_offset:x_offset+w_redim] = chiffre_redim
                img_finale = fond_noir
            else:
                img_finale = np.zeros((28, 28), dtype=np.uint8)
        else:
            # Case vide ou indétectable
            img_finale = np.zeros((28, 28), dtype=np.uint8)

        # Extraction du label pour le tri des dossiers
        parties_nom = nom_case.split('-')
        label_attendu = parties_nom[-1] 
        
        if label_attendu.isdigit():
            dossier_label = os.path.join(dossier_sauvegarde, label_attendu)
            os.makedirs(dossier_label, exist_ok=True)
            chemin_fichier = os.path.join(dossier_label, f"{nom_case}.png")
            cv2.imwrite(chemin_fichier, img_finale)
        else:
            chemin_fichier = os.path.join(dossier_sauvegarde, f"{nom_case}.png")
            cv2.imwrite(chemin_fichier, img_finale)