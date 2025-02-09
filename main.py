import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import os
import cv2
from scipy.ndimage import gaussian_filter, sobel
from skimage import color, io

dossier = 'C:/sportsmot_publish/dataset/foot1'

def egalise_histo(image):
    """
    Égalise l'histogramme de l'image (prétraitement).
    """
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    egalise = cv2.equalizeHist(gris)
    return egalise

def rectangle(image, largeur, longueur):
    """
    Initialise le contour sous la forme d'un rectangle centré au milieu de l'image.
    """
    img_h, img_w = image.shape[:2]
    centre_x, centre_y = img_w // 2, img_h // 2
    haut_gauche = (centre_x - largeur // 2, centre_y - longueur // 2)
    bas_droite = (centre_x + largeur // 2, centre_y + longueur // 2)
    cv2.rectangle(image, haut_gauche, bas_droite, color=(0, 255, 0), thickness=2)
    return image

def ellipse(centre, rx, ry, n):
    """
    Initialise le contour sous la forme d'une ellipse.
    """
    resx = np.zeros(n)
    resy = np.zeros(n)
    for i in range(n):
        resx[i] = centre[0] + rx * np.sin(2 * np.pi * i / n)
        resy[i] = centre[1] + ry * np.cos(2 * np.pi * i / n)
    return resx, resy

def coord_objet(X, Y):
    """
    Retourne les coordonnées moyennes de l'objet.
    """
    n = len(X)
    return sum(X) / n, sum(Y) / n

def dans_image(X, Y, img):
    """
    Vérifie si chaque point de coordonnées (x,y) est bien dans l'image.
    """
    for i in range(len(X)):
        Y[i] = max(0, min(Y[i], img.shape[0] - 1))
        X[i] = max(0, min(X[i], img.shape[1] - 1))
    return X, Y

def creer_M(alpha, beta, n):
    """
    Crée la matrice M pour la dérivation discrète.
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
    return alpha * M2 - beta * M4

def Fx(X, Y, img, lambd, grad2x):
    X, Y = dans_image(X, Y, img)
    return grad2x[Y.round().astype(int), X.round().astype(int)] * lambd

def Fy(X, Y, img, lambd, grad2y):
    X, Y = dans_image(X, Y, img)
    return grad2y[Y.round().astype(int), X.round().astype(int)] * lambd

def evolution_contour(img, X, Y, alpha, beta, gamma, n_iters, lambd, Fnorm, sigma):
    """
    Fait évoluer le contour actif selon les forces internes et externes.
    """
    n = len(X)
    A = creer_M(alpha, beta, n)
    InvM = np.linalg.inv(np.eye(n) - gamma * A)
    im_traitee = gaussian_filter(img, sigma)
    grad_x = sobel(im_traitee, axis=0)
    grad_y = sobel(im_traitee, axis=1)
    norme_grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_x2 = sobel(norme_grad, axis=0)
    grad_y2 = sobel(norme_grad, axis=1)
    X_normal, Y_normal = np.zeros(n), np.zeros(n)
    for _ in range(n_iters):
        for j in range(n-1):
            dX, dY = X[j+1] - X[j], Y[j+1] - Y[j]
            norme = np.sqrt(dX**2 + dY**2)
            X_normal[j], Y_normal[j] = -dX / norme, -dY / norme
        X = np.dot(InvM, X - gamma * (Fx(X, Y, img, lambd, grad_x2) + Fnorm * X_normal))
        Y = np.dot(InvM, Y - gamma * (Fy(X, Y, img, lambd, grad_y2) + Fnorm * Y_normal))
    return X, Y

def suivi_final(Coord_init, rx, ry, n, alpha, beta, gamma, n_iters, lambd, Fnorm, sigma):
    """
    Suivi d'objet basé sur contours actifs.
    """
    contour = []
    im_init = mimg.imread(os.path.join(dossier, os.listdir(dossier)[0]))
    for nom in os.listdir(dossier):
        chemin = os.path.join(dossier, nom)
        img = mimg.imread(chemin)
        img = egalise_histo(img)
        img = gaussian_filter(img, sigma)
        for x0, y0 in Coord_init:
            X, Y = ellipse((x0, y0), rx, ry, n)
            contour.append((X, Y))
        contour_init = contour.copy()
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, im_init.shape[1])
        ax.set_ylim(im_init.shape[0], 0)
        for i in range(len(contour_init)):
            contour[i] = evolution_contour(img, contour[i][0], contour[i][1], alpha, beta, gamma, n_iters, lambd, Fnorm, sigma)
            ax.plot(contour_init[i][0], contour_init[i][1], c='green', lw=2)
            ax.plot(contour[i][0], contour[i][1], c='red', lw=2)
            plt.fill(contour[i][0], contour[i][1], c='red', alpha=0.5)
            x_obj, y_obj = coord_objet(contour[i][0], contour[i][1])
            Coord_init[i] = (x_obj, y_obj)
        plt.show()
