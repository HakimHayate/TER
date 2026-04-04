import json
import cv2
import numpy as np
from utils import read_Qrcode ,filter_valid_markers, mm_to_pixel_list,transform_rects , save_boxes, export_processed_images
import configuration as cf
import json
import os

class Preprocess:
    """
    Classe de moteur de traitement d'images pour le recalage et le découpage de formulaires.
    
    Centralise les paramètres de configuration, les ancres de référence (src) 
    et les métadonnées du template JSON pour optimiser les performances de calcul.

    Attributes:
        src (list): Points d'ancrage théoriques (en mm) pour l'homographie.
        kernel (numpy.ndarray): Élément structurant pour les opérations morphologiques.
        data (dict): Contenu du template JSON chargé en mémoire.
    """
    def __init__(self, src):
        self.src = src
        self.kernel = np.ones((3, 3), np.uint8)
        with open(cf.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        

    def draw_boxes(self,pages,coordinates): 
        """
        Dessine des rectangles de contrôle sur une copie des pages pour la vérification visuelle.
        
        Permet de valider graphiquement l'alignement des zones de saisie après 
        transformation, en superposant des cadres rouges (BGR: 0,0,255) sur les images.

        Args:
            pages (list): Liste des images originales (numpy.ndarray).
            coordinates (list): Liste de listes contenant les coordonnées (xmin, ymin, xmax, ymax) 
                                pour chaque page.

        Returns:
            list: Une nouvelle liste d'images annotées, sans modifier les images sources.
        """
        result = list() 
        for cordinate,page in zip(coordinates,pages) : 
            img = page.copy() 
            for (xmin, ymin, xmax, ymax) in cordinate:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            result.append(img) 
        return result

    def get_streaming_batches(self, img_generator, rects_mm, margin=7, batch_size=512, save_dir=None, start_page_idx=0):
        """
        Orchestre le pipeline de traitement avec support du multiprocessing et sauvegarde de dataset.
        
        Cette méthode transforme les pages PDF en lots (batches) prêts pour l'IA tout en 
        permettant l'archivage physique des captures sur le disque pour le fine-tuning. 
        L'indexation des pages est synchronisée pour éviter les conflits d'écriture 
        entre plusieurs processus.

        Args:
            img_generator (generator): Flux d'images (numpy arrays) provenant du PDF.
            rects_mm (list): Coordonnées théoriques des cases en millimètres.
            margin (int): Marge de sécurité en pixels ajoutée autour de chaque découpe (default 7).
            batch_size (int): Nombre d'images par lot pour l'inférence (default 512).
            save_dir (str, optional): Dossier racine pour sauvegarder les images extraites. 
                                    Si None, la sauvegarde est désactivée.
            start_page_idx (int): Index de départ pour le nommage des dossiers de pages. 
                                Crucial pour la parallélisation (default 0).

        Yields:
            list: Un lot (batch) de tuples (image_MNIST, identifiant_case).
        """
        page_idx = start_page_idx
        for page_img in img_generator:
            json_keys = iter(self.data.keys())
            page_img = cv2.GaussianBlur(page_img, (3, 3), 0)
            # --- [ Logique Homographie existante ] ---
            dst = read_Qrcode(page_img)  
            src_px = mm_to_pixel_list(self.src, page_img)
            src_clean, dst_clean = filter_valid_markers(src_px, dst)
            if len(src_clean) < 4: continue
            
            H, _ = cv2.findHomography(np.array(src_clean, dtype=np.float32), 
                                    np.array(dst_clean, dtype=np.float32))
            
            rects_px = mm_to_pixel_list(rects_mm, page_img)
            rects_transformed = transform_rects(rects_px, H)
            
            
            # --- PRÉPARATION DOSSIER PAGE (Si option activée) ---
            current_page_dir = None
            if save_dir:
                current_page_dir = os.path.join(save_dir, str(page_idx))
                os.makedirs(current_page_dir, exist_ok=True)

            current_batch = []
            h_img, w_img = page_img.shape[:2]
            
            for (xmin, ymin, xmax, ymax) in rects_transformed:
                xmin_m, ymin_m = max(0, xmin - margin), max(0, ymin - margin)
                xmax_m, ymax_m = min(w_img, xmax + margin), min(h_img, ymax + margin)

                crop = page_img[ymin_m:ymax_m, xmin_m:xmax_m]
                crop_for_ia = self.to_mnist_format(crop) 
                
                try:
                    crop_label = next(json_keys)
                    
                    # --- SAUVEGARDE PHYSIQUE (Optionnelle) ---
                    if current_page_dir:
                        save_boxes(current_page_dir, crop_for_ia, crop_label)

                    current_batch.append((crop_for_ia, crop_label))
                except StopIteration: break
                
                if len(current_batch) == batch_size:
                    yield current_batch
                    current_batch = []
            
            if current_batch:
                yield current_batch
                
            page_idx += 1 

    def to_mnist_format(self,img, out_size=28):
        """
        Normalise une zone découpée pour la rendre compatible avec un modèle de type MNIST.
        
        Le traitement inclut la conversion en niveaux de gris, une binarisation inverse 
        (chiffre blanc sur fond noir), un recentrage par masse, un rognage des bordures 
        pour éliminer les cadres parasites, et un redimensionnement final.

        Args:
            img (numpy.ndarray): L'image brute de la case (crop) extraite du formulaire.
            out_size (int): La taille de sortie carrée (par défaut 28x28 pixels).

        Returns:
            numpy.ndarray: Une image binaire normalisée, centrée et redimensionnée, 
                        prête pour l'inférence par le réseau de neurones.
        """
        remove_padding = 11

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, img = cv2.threshold(img, cf.threshold, 255,cv2.THRESH_BINARY_INV )

        img = self.center_by_mass(img)
        cropped = img[remove_padding:-remove_padding, remove_padding:-remove_padding]
        digit = cv2.resize(cropped,(28,28))
        
        return digit
    
    def center_by_mass(self, img):
        """
        Recentrer dynamiquement le contenu de l'image en fonction de son centre de masse.
        
        Utilise les moments d'image (cv2.moments) sur une version érodée de l'image pour 
        identifier le cœur du tracé (ignorant les traits fins/bruit) et applique une 
        transformation affine pour centrer le chiffre dans le cadre.

        Args:
            img (numpy.ndarray): Image de la case (crop) en niveaux de gris.

        Returns:
            numpy.ndarray: Image translatée (shifted) où le contenu principal est 
                        parfaitement centré pour faciliter la reconnaissance (OCR/IA).
        """
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        else:
            img_uint8 = img

        h, w = img_uint8.shape

        kernel = np.ones((2, 2), np.uint8)
        temp_img = cv2.erode(img_uint8, kernel, iterations=1)

        M = cv2.moments(temp_img)
        
        if M["m00"] == 0:
            M = cv2.moments(img_uint8)

        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            shift_x = (w / 2.0) - cx
            shift_y = (h / 2.0) - cy
            
            matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            shifted = cv2.warpAffine(
                img, 
                matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR, 
                borderValue=0
            )
            return shifted
        else:
            return img