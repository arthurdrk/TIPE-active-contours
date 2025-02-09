import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from skimage import color, io
import os

def egalise_histo(image):
    """
    Égalise l'histogramme de l'image pour améliorer le contraste.
    """
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gris)

def rectangle(image, largeur, longueur):
    """
    Initialise un contour rectangulaire centré dans l'image.
    """
    img_h, img_w = image.shape[:2]
    centre_x, centre_y = img_w // 2, img_h // 2
    haut_gauche = (centre_x - largeur//2, centre_y - longueur // 2)
    bas_droite = (centre_x + largeur // 2, centre_y + longueur // 2)
    cv2.rectangle(image, haut_gauche, bas_droite, color=(0, 255, 0), thickness=2)
    return image

def ellipse(centre, rx, ry, n):
    """
    Crée une ellipse en fonction des coordonnées du centre et des axes.
    """
    resx = np.array([centre[0] + rx * np.sin(2 * np.pi * i / n) for i in range(n)])
    resy = np.array([centre[1] + ry * np.cos(2 * np.pi * i / n) for i in range(n)])
    return resx, resy

def evolution_contour(img, X, Y, alpha, beta, gamma, n_iters, lambd, Fnorm, sigma):
    """
    Algorithme de suivi du contour actif.
    """
    n = len(X)
    A = np.eye(n) - gamma * (alpha * np.eye(n) - beta * np.eye(n))
    InvM = np.linalg.inv(A)
    im_traitee = gaussian_filter(img, sigma)
    grad_x = sobel(im_traitee, axis=0)
    grad_y = sobel(im_traitee, axis=1)
    norme_grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_x2 = sobel(norme_grad, axis=0)
    grad_y2 = sobel(norme_grad, axis=1)
    for _ in range(n_iters):
        Xiter = np.dot(InvM, X - gamma * (grad_x2 + Fnorm))
        Yiter = np.dot(InvM, Y - gamma * (grad_y2 + Fnorm))
        X, Y = Xiter.copy(), Yiter.copy()
    return X, Y

# Chargement de l'image de test
img = io.imread('sample_image.png')
img = color.rgb2gray(img)
img = egalise_histo(img)

# Paramètres du contour
X, Y = ellipse((img.shape[1]//2, img.shape[0]//2), 50, 30, 100)

# Exécution du suivi d'objet
X_final, Y_final = evolution_contour(img, X, Y, alpha=0.1, beta=0.1, gamma=0.1, n_iters=100, lambd=0.5, Fnorm=0.5, sigma=1)

# Affichage du résultat
plt.figure()
plt.imshow(img, cmap='gray')
plt.plot(X, Y, 'r--')
plt.plot(X_final, Y_final, 'g-')
plt.show()
