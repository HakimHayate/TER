import random
import numpy as np
import torch

class DataAugmentation :
    """
    Simulateur de bruits et de distorsions réalistes pour formulaires scannés.
    
    Cette classe enrichit le dataset d'entraînement en ajoutant des artefacts 
    typiques de la numérisation : décalages horizontaux, traits de bordure de cases 
    parasites (bruit rectangulaire) et variations d'intensité (gris).

    Attributes:
        kernel (numpy.ndarray): Élément structurant pour d'éventuelles opérations morphologiques.
    """
    def __init__(self):
        self.kernel = np.ones((3,3),np.uint8)

    def gris(self) :
        return np.random.choice([0, 0.3,0.15], p=[0.4, 0.35, 0.25])


    def start(self):
        return np.random.choice([0,1,2,3,4,5,6], p=[0.7, 0.1,0.05,0.05,0.05,0.03,0.02])


    def end(self):
        return 28 - np.random.choice([0,1,2,3,4,5,6], p=[0.7, 0.1,0.05,0.05,0.05,0.03,0.02])



    def horizontal_shift(self, img, shift):
        """
        Applique une translation horizontale au tenseur de l'image.
        Simule un mauvais alignement du chiffre dans sa case.
        """
        c, h, w = img.shape
        result = torch.zeros_like(img)
        if shift > 0: # droite
            result[:, :, shift:] = img[:, :, :w-shift]

        elif shift < 0: # gauche
            result[:, :, :w+shift] = img[:, :, -shift:]

        else:
            result = img

        return result



    def rectangle_noise(self,img_org) :
        """
        Génère du bruit de structure (lignes de bordure) sur les contours de l'image.
        
        Cette méthode simule le cas où le découpage de la case (crop) n'est pas parfait 
        et laisse apparaître un morceau du cadre du formulaire ou un trait voisin. 
        Le bruit est appliqué de manière probabiliste avec des intensités de gris variables.

        Args:
            img_org (torch.Tensor): Image MNIST originale (Tenseur [C, H, W]).

        Returns:
            torch.Tensor: Image augmentée avec décalage et potentiels traits parasites.
        """
        img = img_org.clone()
        shifts = [-7,-6,-5,-4, -3, -2, -1, 0, 1, 2, 3, 4,5,6]
        p = [0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.1,0.05,0.04,0.03,0.02,0.01]
        shift = np.random.choice(shifts, p=p)
        img = self.horizontal_shift(img, shift)

        if random.random() > 0.9:

            thickness = 1
            val = 1
            h, w = img.shape[-2], img.shape[-1]

            if random.random() > 0.8:
                decalage = np.random.choice([0,1,2], p=[0.82, 0.15, 0.03])
                img[..., decalage:decalage+thickness,self.start() : self.end()] = val - self.gris()

            elif random.random() > 0.1:
                decalage = np.random.choice([0,1,2], p=[0.82, 0.15, 0.03])
                img[..., self.start() : self.end(), decalage:decalage+thickness] = val - self.gris()

            if random.random() > 0.8:
                decalage = np.random.choice([0,1,2], p=[0.82, 0.15, 0.03])
                img[..., self.start() : self.end(), w-decalage-thickness:w-decalage] = val - self.gris()

            elif random.random() > 0.8:
                decalage = np.random.choice([0,1,2], p=[0.82, 0.15, 0.03])
                img[..., h-decalage-thickness:h-decalage, self.start() : self.end()] = val - self.gris()


        return img