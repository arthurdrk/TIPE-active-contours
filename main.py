from scipy.ndimage import gaussian_filter, sobel
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
import matplotlib.image as mimg
import os
import cv2

dossier = 'C:/sportsmot_publish/dataset/foot1'

def egalise_histo(image):
    """
    Egalise l'histogramme de l'image (prétraitement).
    """
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    egalise = cv2.equalizeHist(gris)
    return egalise

def rectangle(image, largeur, longueur):
    """
    Initialise le contour sous la forme d'un rectangle centré au milieu de l'image.
    Renvoie deux tableaux :
    X : coordonnées x des points du rectangle
    Y : coordonnées y des points du rectangle
    """
    img_h, img_w = image.shape[:2]
    centre_x, centre_y = img_w // 2, img_h // 2
    x_min = centre_x - largeur // 2
    x_max = centre_x + largeur // 2
    y_min = centre_y - longueur // 2
    y_max = centre_y + longueur // 2

    # On définit le contour du rectangle en reprenant le premier point pour fermer la boucle
    X = np.array([x_min, x_max, x_max, x_min, x_min])
    Y = np.array([y_min, y_min, y_max, y_max, y_min])
    return X, Y



def ellipse(centre, rx, ry, n):
    """
    Initialise le contour sous la forme d'une ellipse dont la position du centre et
    les longueurs du petit et du grand axe sont données.
    """
    resx = np.zeros(n)
    resy = np.zeros(n)
    for i in range(n):
        resx[i] = centre[0] + rx * np.sin(2 * np.pi * i / n)
        resy[i] = centre[1] + ry * np.cos(2 * np.pi * i / n)
    return resx, resy

def coord_objet(X, Y):
    """
    Entrée : Coordonnées des points du contour final de l'image précédente.
    Sortie : Coordonnées de l'objet.
    """
    n = len(X)
    return sum(X) / n, sum(Y) / n

def dans_image(X, Y, img):
    """
    Vérifie si chaque point de coordonnées (x,y) est bien dans l'image.
    Sinon, renvoie le point sur le bord de l'image le plus proche.
    """
    for i in range(len(X)):
        if Y[i] < 0:
            Y[i] = 0
        if Y[i] > img.shape[0] - 1:
            Y[i] = img.shape[0] - 1
        if X[i] < 0:
            X[i] = 0
        if X[i] > img.shape[1] - 1:
            X[i] = img.shape[1] - 1
    return X, Y


def creer_M(alpha, beta, n):
    """
    Crée la matrice M correspondant à la dérivation discrète de v obtenue par la méthode
    des différences finies pour les paramètres alpha et beta.
    """
    assert n > 5
    M2 = np.array([[0] * n for _ in range(n)])
    M4 = np.array([[0] * n for _ in range(n)])
    for i in range(n):
        M2[i][i] = -2
        M4[i][i] = 6
        if i > 0:
            M2[i][i-1] = 1
            M2[i-1][i] = 1
            M4[i][i-1] = -4
            M4[i-1][i] = -4
        if i > 1:
            M4[i][i-2] = 1
            M4[i-2][i] = 1
        if i < n-1:
            M4[i][i+1] = -4
            M4[i+1][i] = -4
        if i < n-2:
            M4[i][i+2] = 1
            M4[i+2][i] = 1
    M2[0][-1] = 1
    M2[n-1][0] = 1
    M4[0][n-2] = 1
    M4[0][n-1] = -4
    M4[1][n-1] = 1
    M4[n-2][0] = 1
    M4[n-1][0] = -4
    M4[n-1][1] = 1
    return alpha * M2 - beta * M4


def Fx(X, Y, img, lambd, grad2x):
    """
    Entrée : Coordonnées X,Y des points du contour, l'image à segmenter, lambda
    le poids de l'énergie externe, le gradient du module du gradient de l'intensité
    suivant x.
    Sortie : terme de force Fx correspondant à l'énergie externe (liste de taille n).
    """
    X, Y = dans_image(X, Y, img)
    return grad2x[(Y.round().astype(int), X.round().astype(int))] * lambd


def Fy(X, Y, img, lambd, grad2y):
    """
    Entrée : Coordonnées X,Y des points du contour, l'image à segmenter, lambda
    le poids de l'énergie externe, le gradient du module du gradient de l'intensité
    suivant y.
    Sortie : terme de force Fy correspondant à l'énergie externe (liste de taille n).
    """
    X, Y = dans_image(X, Y, img)
    return grad2y[(Y.round().astype(int), X.round().astype(int))] * lambd


def evolution_contour(img, X, Y, alpha, beta, gamma, n_iters, lambd, Fnorm, sigma):
    """
    Entrée : image, coordonnées initiales du contour (X,Y),
    paramètre alpha (élasticité), beta (rigidité), gamma (pas temporel), n_iters
    (nombre d'itérations), lambda (poids de l'énergie liée à l'image),
    Fnorm (force normale), sigma (écart-type du flou gaussien).
    Sortie : Liste finale des positions X,Y des points du contour actif.
    """
    n = len(X)
    A = creer_M(alpha, beta, n)
    M = np.eye(n) - gamma * A
    InvM = np.linalg.inv(M)
    # Définition des vecteurs normaux pour la force normale
    X_normal = np.zeros(n)
    Y_normal = np.zeros(n)
    im_traitee = gaussian_filter(img, sigma)
    # Calcul du gradient de l'image
    grad_x = sobel(im_traitee, axis=0)
    grad_y = sobel(im_traitee, axis=1)
    norme_grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_x2 = sobel(norme_grad, axis=0)
    grad_y2 = sobel(norme_grad, axis=1)
    
    # Configuration de l'affichage interactif
    plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    # Affichage du contour initial en rouge
    ax.plot(X, Y, 'r-', lw=2, label='Contour initial')
    plt.draw()
    plt.pause(0.5)
    
    for i in range(n_iters):
        # Calcul des vecteurs normaux au contour
        for j in range(n-1):
            dX = X[j+1] - X[j]
            dY = Y[j+1] - Y[j]
            norme = np.sqrt(dX**2 + dY**2)
            X_normal[j] = -dX / norme
            Y_normal[j] = -dY / norme
        # Evolution suivant le schéma itératif
        Xiter = np.dot(InvM, X - gamma * (Fx(X, Y, img, lambd, grad_x2) + Fnorm * X_normal))
        Yiter = np.dot(InvM, Y - gamma * (Fy(X, Y, img, lambd, grad_y2) + Fnorm * Y_normal))
        X, Y = Xiter.copy(), Yiter.copy()
        
        # Affichage du contour intermédiaire ou final
        if i < n_iters - 1:
            ax.plot(X, Y, 'g-', lw=1)  # contour intermédiaire en vert
        else:
            ax.plot(X, Y, 'y-', lw=2, label='Contour final')  # contour final en jaune
        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()
    plt.show()
    return (X, Y)



def suivi_final(Coord_init, rx, ry, n, alpha, beta, gamma, n_iters, lambd, Fnorm, sigma):
    contour = []
    # Création du contour initial pour chaque objet.
    im_init = mimg.imread(os.path.join(dossier, os.listdir(dossier)[0]))
    # Parcours de toutes les images de la vidéo.
    for nom in os.listdir(dossier):
        chemin = os.path.join(dossier, nom)
        img = mimg.imread(chemin)
        img = egalise_histo(img)
        img = gaussian_filter(img, sigma)
        for x0, y0 in Coord_init:
            X, Y = ellipse((x0, y0), rx, ry, n)
            contour.append((X, Y))
    contour_init = contour.copy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, im_init.shape[1])
    ax.set_ylim(im_init.shape[0], 0)
    for i in range(len(contour_init)):
        contour[i] = evolution_contour(img, contour[i][0], contour[i][1], alpha, beta, gamma, n_iters, lambd, Fnorm, sigma)
        # Trace le contour initial.
        ax.plot(contour_init[i][0], contour_init[i][1], c='green', lw=2)
        # Trace le contour final.
        ax.plot(contour[i][0], contour[i][1], c='red', lw=2)
        plt.fill(contour[i][0], contour[i][1], c='red', alpha=0.5)
        # Actualise les coordonnées de l'objet en mouvement.
        x_obj, y_obj = coord_objet(contour[i][0], contour[i][1])
        Coord_init[i] = (x_obj, y_obj)
    plt.show()
