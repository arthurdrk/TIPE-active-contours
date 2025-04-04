from scipy.ndimage import gaussian_filter, sobel
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
import matplotlib.image as mimg
import os
import cv2
from typing import Tuple, List

class ImageProcessor:
    """Classe pour le prétraitement des images.

    Cette classe fournit des méthodes pour le prétraitement des images,
    comme l'égalisation d'histogramme.
    """

    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """Égalise l'histogramme de l'image.

        Parameters
        ----------
        image : np.ndarray
            Image d'entrée (BGR ou niveaux de gris).

        Returns
        -------
        np.ndarray
            Image avec histogramme égalisé en niveaux de gris.
        """
        # Vérifier si l'image est déjà en niveaux de gris
        if len(image.shape) == 2:
            gray = image
        else:
            # Convertir en niveaux de gris si l'image est en couleur
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # S'assurer que l'image est en uint8
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
            
        equalized = cv2.equalizeHist(gray)
        return equalized

class ContourInitializer:
    """Classe pour l'initialisation des contours.

    Cette classe fournit des méthodes pour créer différents types de contours
    initiaux (rectangle, ellipse).
    """

    @staticmethod
    def create_ellipse(center: Tuple[float, float], rx: float, ry: float, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Crée un contour elliptique.

        Parameters
        ----------
        center : Tuple[float, float]
            Coordonnées du centre de l'ellipse.
        rx : float
            Rayon horizontal de l'ellipse.
        ry : float
            Rayon vertical de l'ellipse.
        n_points : int
            Nombre de points pour discrétiser le contour.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Coordonnées X et Y des points du contour.
        """
        X = np.zeros(n_points)
        Y = np.zeros(n_points)
        for i in range(n_points):
            X[i] = center[0] + rx * np.sin(2 * np.pi * i / n_points)
            Y[i] = center[1] + ry * np.cos(2 * np.pi * i / n_points)
        return X, Y

class ActiveContour:
    """Classe implémentant l'algorithme des contours actifs.

    Cette classe gère l'évolution du contour actif selon les forces internes
    et externes.
    """

    def __init__(self, alpha: float, beta: float, gamma: float, lambda_: float,
                 f_norm: float, sigma: float):
        """Initialise les paramètres du contour actif.

        Parameters
        ----------
        alpha : float
            Paramètre d'élasticité.
        beta : float
            Paramètre de rigidité.
        gamma : float
            Pas temporel.
        lambda_ : float
            Poids de l'énergie liée à l'image.
        f_norm : float
            Force normale.
        sigma : float
            Écart-type du flou gaussien.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.f_norm = f_norm
        self.sigma = sigma

    def _create_matrix_M(self, n: int) -> np.ndarray:
        """Crée la matrice M pour la dérivation discrète.

        Parameters
        ----------
        n : int
            Nombre de points du contour.

        Returns
        -------
        np.ndarray
            Matrice M pour l'évolution du contour.
        """
        assert n > 5
        M2 = np.zeros((n, n))
        M4 = np.zeros((n, n))

        for i in range(n):
            M2[i, i] = -2
            M4[i, i] = 6
            if i > 0:
                M2[i, i-1] = M2[i-1, i] = 1
                M4[i, i-1] = M4[i-1, i] = -4
            if i > 1:
                M4[i, i-2] = M4[i-2, i] = 1
            if i < n-1:
                M4[i, i+1] = M4[i+1, i] = -4
            if i < n-2:
                M4[i, i+2] = M4[i+2, i] = 1

        M2[0, -1] = M2[-1, 0] = 1
        M4[0, -2] = M4[-2, 0] = 1
        M4[0, -1] = M4[-1, 0] = -4
        M4[1, -1] = M4[-1, 1] = 1

        return self.alpha * M2 - self.beta * M4

    def _compute_forces(self, X: np.ndarray, Y: np.ndarray, img: np.ndarray,
                       grad_x: np.ndarray, grad_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule les forces internes et externes.

        Parameters
        ----------
        X : np.ndarray
            Coordonnées X des points du contour.
        Y : np.ndarray
            Coordonnées Y des points du contour.
        img : np.ndarray
            Image à segmenter.
        grad_x : np.ndarray
            Gradient suivant x.
        grad_y : np.ndarray
            Gradient suivant y.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Forces Fx et Fy.
        """
        X, Y = self._keep_in_bounds(X, Y, img)
        Fx = grad_x[Y.round().astype(int), X.round().astype(int)] * self.lambda_
        Fy = grad_y[Y.round().astype(int), X.round().astype(int)] * self.lambda_
        return Fx, Fy

    @staticmethod
    def _keep_in_bounds(X: np.ndarray, Y: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Maintient les points du contour dans les limites de l'image.

        Parameters
        ----------
        X : np.ndarray
            Coordonnées X des points.
        Y : np.ndarray
            Coordonnées Y des points.
        img : np.ndarray
            Image de référence.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Coordonnées ajustées.
        """
        Y = np.clip(Y, 0, img.shape[0] - 1)
        X = np.clip(X, 0, img.shape[1] - 1)
        return X, Y

    def evolve(self, img: np.ndarray, X: np.ndarray, Y: np.ndarray,
              n_iterations: int, show_evolution: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Fait évoluer le contour actif.

        Parameters
        ----------
        img : np.ndarray
            Image à segmenter.
        X : np.ndarray
            Coordonnées X initiales.
        Y : np.ndarray
            Coordonnées Y initiales.
        n_iterations : int
            Nombre d'itérations.
        show_evolution : bool, optional
            Affiche l'évolution du contour, by default True

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Coordonnées finales du contour.
        """
        n = len(X)
        A = self._create_matrix_M(n)
        M = np.eye(n) - self.gamma * A
        InvM = np.linalg.inv(M)

        im_processed = gaussian_filter(img.astype(np.float32), self.sigma)
        grad_x = sobel(im_processed, axis=0)
        grad_y = sobel(im_processed, axis=1)
        norm_grad = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
        grad_x2 = sobel(norm_grad, axis=0)
        grad_y2 = sobel(norm_grad, axis=1)

        if show_evolution:
            plt.ion()
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.plot(X, Y, 'r-', lw=2, label='Contour initial')
            plt.draw()
            plt.pause(0.5)

        for i in range(n_iterations):
            X_normal = np.zeros(n)
            Y_normal = np.zeros(n)

            for j in range(n-1):
                dX = X[j+1] - X[j]
                dY = Y[j+1] - Y[j]
                norm = np.sqrt(dX**2 + dY**2)
                X_normal[j] = -dX / norm
                Y_normal[j] = -dY / norm

            Fx, Fy = self._compute_forces(X, Y, img, grad_x2, grad_y2)

            X_new = np.dot(InvM, X - self.gamma * (Fx + self.f_norm * X_normal))
            Y_new = np.dot(InvM, Y - self.gamma * (Fy + self.f_norm * Y_normal))
            X, Y = X_new.copy(), Y_new.copy()

            if show_evolution:
                if i < n_iterations - 1:
                    ax.plot(X, Y, 'g-', lw=1)
                else:
                    ax.plot(X, Y, 'y-', lw=2, label='Contour final')
                plt.draw()
                plt.pause(0.1)

        if show_evolution:
            plt.ioff()
            plt.show()

        return X, Y

class ContourTracker:
    """Classe pour le suivi de contours dans une séquence d'images.

    Cette classe gère le suivi de plusieurs contours à travers une séquence
    d'images.
    """

    def __init__(self):
        """Initialise le tracker de contours."""
        self.image_processor = ImageProcessor()
        self.contour_initializer = ContourInitializer()

    def track_single_image(self, image_path: str, initial_coords: List[Tuple[float, float]],
                          rx: float, ry: float, n_points: int, active_contour: ActiveContour,
                          n_iterations: int) -> None:
        """Suit les contours sur une seule image.

        Parameters
        ----------
        image_path : str
            Chemin vers l'image.
        initial_coords : List[Tuple[float, float]]
            Liste des coordonnées initiales des centres des contours.
        rx : float
            Rayon horizontal des ellipses initiales.
        ry : float
            Rayon vertical des ellipses initiales.
        n_points : int
            Nombre de points pour discrétiser les contours.
        active_contour : ActiveContour
            Instance de ActiveContour pour l'évolution des contours.
        n_iterations : int
            Nombre d'itérations pour l'évolution des contours.
        """
        img = mimg.imread(image_path)
        img_gray = self.image_processor.equalize_histogram(img)
        img_gray = gaussian_filter(img_gray, active_contour.sigma)

        contours = []
        for x0, y0 in initial_coords:
            X, Y = self.contour_initializer.create_ellipse((x0, y0), rx, ry, n_points)
            contours.append((X, Y))
            
        initial_contours = contours.copy()
        
        # Configuration de l'affichage
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)

        # Affichage des paramètres dans un carré en haut à droite
        param_text = f"Paramètres:\nα={active_contour.alpha}\nβ={active_contour.beta}\nγ={active_contour.gamma}\nλ={active_contour.lambda_}\nF={active_contour.f_norm}\nσ={active_contour.sigma}"
        ax.text(0.98, 0.98, param_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.8))

        for i in range(len(initial_contours)):
            contours[i] = active_contour.evolve(img_gray, contours[i][0], contours[i][1],
                                              n_iterations=n_iterations, show_evolution=False)
            ax.plot(initial_contours[i][0], initial_contours[i][1], c='green', lw=2)
            ax.plot(contours[i][0], contours[i][1], c='red', lw=2)
            plt.fill(contours[i][0], contours[i][1], c='red', alpha=0.5)

            x_obj, y_obj = np.mean(contours[i][0]), np.mean(contours[i][1])
            initial_coords[i] = (x_obj, y_obj)

        plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    tracker = ContourTracker()

    # # 1. Paramètres pour le trèfle
    # trefle_contour = ActiveContour(
    #     alpha=70,      # paramètre d'élasticité
    #     beta=0.001,    # paramètre de rigidité
    #     gamma=0.005,   # pas temporel
    #     lambda_=1,     # poids de l'énergie liée à l'image
    #     f_norm=0,      # force normale
    #     sigma=1.0      # écart-type du flou gaussien
    # )

    # # 2. Paramètres pour l'étoile
    # etoile_contour = ActiveContour(
    #     alpha=50,      # paramètre d'élasticité
    #     beta=0.1,      # paramètre de rigidité
    #     gamma=0.01,    # pas temporel
    #     lambda_=1.5,   # poids de l'énergie liée à l'image
    #     f_norm=0.1,    # force normale
    #     sigma=1.5      # écart-type du flou gaussien
    # )

    # 3. Paramètres pour le pique
    pique_contour = ActiveContour(
        alpha=3,      # paramètre d'élasticité
        beta=0.01,      # paramètre de rigidité
        gamma=0.02,    # pas temporel
        lambda_=0.1,   # poids de l'énergie liée à l'image
        f_norm=0,    # force normale
        sigma=1.0      # écart-type du flou gaussien
    )

    # # 4. Paramètres pour le vase
    # vase_contour = ActiveContour(
    #     alpha=40,      # paramètre d'élasticité
    #     beta=0.05,     # paramètre de rigidité
    #     gamma=0.01,    # pas temporel
    #     lambda_=1.2,   # poids de l'énergie liée à l'image
    #     f_norm=0.15,   # force normale
    #     sigma=1.8      # écart-type du flou gaussien
    # )

    # # Traitement du trèfle
    # tracker.track_single_image(
    #     image_path='./img/trefle.jpg',
    #     initial_coords=[(160, 150)],
    #     rx=140,
    #     ry=140,
    #     n_points=100,
    #     active_contour=trefle_contour,
    #     n_iterations=10000
    # )

    # # Traitement de l'étoile
    # tracker.track_single_image(
    #     image_path='./img/etoile.png',
    #     initial_coords=[(320, 320)],
    #     rx=300,
    #     ry=300,
    #     n_points=150,
    #     active_contour=etoile_contour,
    #     n_iterations=8000
    # )

    # Traitement du pique
    tracker.track_single_image(
        image_path='./img/pique.jpg',
        initial_coords=[(550, 550)],
        rx=500,
        ry=500,
        n_points=100,
        active_contour=pique_contour,
        n_iterations=10000
    )

    # # Traitement du vase
    # tracker.track_single_image(
    #     image_path='./img/vase.tif',
    #     initial_coords=[(150, 150)],
    #     rx=110,
    #     ry=140,
    #     n_points=180,
    #     active_contour=vase_contour,
    #     n_iterations=10000
    # )
