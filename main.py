import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import matplotlib.image as mimg

def create_A(a, b, N):
    """
    Create the coefficient matrix A for snake energy minimization.

    Parameters
    ----------
    a : float
        Alpha parameter controlling tension.
    b : float
        Beta parameter controlling rigidity.
    N : int
        Number of contour points.

    Returns
    -------
    A : ndarray
        N x N coefficient matrix for snake energy calculation.
    """
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N, N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A

def create_external_edge_force_gradients_from_img(img, sigma=30.):
    """
    Compute gradient functions for external edge forces from an image.

    Parameters
    ----------
    img : ndarray
        Input image for force calculation.
    sigma : float, optional
        Sigma for Gaussian smoothing, by default 30.

    Returns
    -------
    fx, fy : callable
        Functions computing x/y components of external edge forces.
    """
    # Convertir l'image en float si elle est booléenne
    img_float = img.astype(float)
    smoothed = gaussian((img_float - img_float.min()) / (img_float.max() - img_float.min()), sigma)
    giy, gix = np.gradient(smoothed)
    gmi = (gix**2 + giy**2)**0.5
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
    ggmiy, ggmix = np.gradient(gmi)

    def fx(x, y):
        x = np.clip(x, 0, img.shape[1]-1).round().astype(int)
        y = np.clip(y, 0, img.shape[0]-1).round().astype(int)
        return ggmix[y, x]

    def fy(x, y):
        x = np.clip(x, 0, img.shape[1]-1).round().astype(int)
        y = np.clip(y, 0, img.shape[0]-1).round().astype(int)
        return ggmiy[y, x]

    return fx, fy

def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    """
    Perform snake contour evolution through energy minimization.

    Parameters
    ----------
    x : ndarray
        Initial x-coordinates of snake contour.
    y : ndarray
        Initial y-coordinates of snake contour.
    a : float
        Tension parameter (alpha).
    b : float
        Rigidity parameter (beta).
    fx : callable
        X-component of external force field.
    fy : callable
        Y-component of external force field.
    gamma : float, optional
        Step size for iteration, by default 0.1.
    n_iters : int, optional
        Number of iterations, by default 10.
    return_all : bool, optional
        Return all iterations if True, by default True.

    Returns
    -------
    list or tuple
        All snake iterations if return_all=True, else final coordinates.
    """
    A = create_A(a, b, x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma * A)
    snakes = []

    for _ in range(n_iters):
        x_new = B @ (x + gamma * fx(x, y))
        y_new = B @ (y + gamma * fy(x, y))
        x, y = x_new.copy(), y_new.copy()
        if return_all:
            snakes.append((x.copy(), y.copy()))

    return snakes if return_all else (x, y)

def process_image(img_path, x_center, y_center, rx, ry, n_points, alpha, beta, gamma, sigma, n_iters):
    """
    Process an image with active contour.
    
    Parameters
    ----------
    img_path : str
        Path to the image file.
    x_center, y_center : float
        Center coordinates for the initial contour.
    rx, ry : float
        Radii for the initial elliptical contour.
    n_points : int
        Number of points to discretize the contour.
    alpha, beta, gamma : float
        Snake parameters.
    sigma : float
        Gaussian smoothing parameter.
    n_iters : int
        Number of iterations for snake evolution.
    """
    # Load image
    original_img = mimg.imread(img_path)
    
    # Create grayscale version for processing
    if len(original_img.shape) == 3:
        gray_img = np.mean(original_img, axis=2)
    else:
        gray_img = original_img.copy()
    
    # Initialize snake contour
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = x_center + rx * np.cos(t)
    y = y_center + ry * np.sin(t)
    
    # Create force fields using grayscale image
    fx, fy = create_external_edge_force_gradients_from_img(gray_img, sigma=sigma)
    
    # Run snake evolution
    snakes = iterate_snake(x, y, alpha, beta, fx, fy, gamma, n_iters)
    
    # Visualization with original color image
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(original_img)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot initial contour
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], c=(0,1,0), lw=2, label='Initial contour')
    
    # Plot intermediate contours
    for i, (sx, sy) in enumerate(snakes):
        if i % 150 == 0:
            ax.plot(np.r_[sx, sx[0]], np.r_[sy, sy[0]], c=(0,0,1), lw=1)
    
    # Plot final contour
    ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], 
            np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2, label='Final contour')
    
    # Add parameters text
    param_text = f"Parameters:\nα={alpha}\nβ={beta}\nγ={gamma}\nσ={sigma}"
    ax.text(0.98, 0.98, param_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.legend()
    plt.title(f"Active Contour on {img_path}")
    plt.show()

if __name__ == "__main__":
    # 1. Process synthetic image
    print("Processing synthetic image...")
    # Generate synthetic image
    x_grid, y_grid = np.mgrid[-4:4:256j, -4:4:256j]
    rad = (x_grid**2 + y_grid**2)**0.5
    theta = np.arctan2(y_grid, x_grid)
    img = (rad <= (2 + np.sin(4*theta))).astype(float) + 0.1 * np.random.randn(256, 256)

    # Initialize snake contour
    t = np.arange(0, 2*np.pi, 0.1)
    x = 128 + 100*np.cos(t)
    y = 128 + 100*np.sin(t)

    # Set parameters
    alpha = 0.001
    beta = 0.01
    gamma = 100
    iterations = 50

    # Create force fields
    fx, fy = create_external_edge_force_gradients_from_img(img, sigma=10)

    # Run snake evolution
    snakes = iterate_snake(x, y, alpha, beta, fx, fy, gamma, iterations)

    # Visualization
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], c=(0,1,0), lw=2)

    for i, (sx, sy) in enumerate(snakes):
        if i % 10 == 0:
            ax.plot(np.r_[sx, sx[0]], np.r_[sy, sy[0]], c=(0,0,1), lw=2)

    ax.plot(np.r_[snakes[-1][0], snakes[-1][0][0]], 
            np.r_[snakes[-1][1], snakes[-1][1][0]], c=(1,0,0), lw=2)
    plt.title("Synthetic Image")
    plt.show()

    # 2. Process trèfle image
    print("Processing trèfle image...")
    process_image(
        img_path='img/trefle.jpg',
        x_center=150,
        y_center=150,
        rx=150,
        ry=150,
        n_points=200,
        alpha=0.03,
        beta=0.1,
        gamma=10,
        sigma=2.0,
        n_iters=10000
    )

    print("Processing pique image...")
    process_image(
        img_path='img/pique.jpg',
        x_center=550,
        y_center=550,
        rx=450,
        ry=450,
        n_points=300,
        alpha=0.03,
        beta=0.001,
        gamma=2,
        sigma=1.0,
        n_iters=16000
    )

    print("Processing etoile image...")
    process_image(
        img_path='img/etoile.png',
        x_center=320,
        y_center=320,
        rx=340,
        ry=340,
        n_points=200,
        alpha=0.01,
        beta=0.001,
        gamma=8,
        sigma=1.0,
        n_iters=16000
    )

    print("Processing vase image...")
    process_image(
        img_path='img/vase.jpg',
        x_center=100,
        y_center=90,
        rx=70,
        ry=70,
        n_points=200,
        alpha=0.001,
        beta=1,
        gamma=30,
        sigma=1.0,
        n_iters=20000
    )