import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
import matplotlib.image as mimg

class ActiveContour:
    def __init__(self, alpha, beta, gamma, sigma, n_iters):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.n_iters = n_iters

    def create_A(self, N):
        row = np.r_[
            -2*self.alpha - 6*self.beta, 
            self.alpha + 4*self.beta,
            -self.beta,
            np.zeros(N-5),
            -self.beta,
            self.alpha + 4*self.beta
        ]
        A = np.zeros((N, N))
        for i in range(N):
            A[i] = np.roll(row, i)
        return A

    def create_external_edge_force_gradients(self, img):
        img_float = img.astype(float)
        smoothed = gaussian((img_float - img_float.min()) / (img_float.max() - img_float.min()), self.sigma)
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

    def iterate(self, x, y, fx, fy):
        A = self.create_A(x.shape[0])
        B = np.linalg.inv(np.eye(x.shape[0]) - self.gamma * A)
        snakes = []

        for _ in range(self.n_iters):
            x_new = B @ (x + self.gamma * fx(x, y))
            y_new = B @ (y + self.gamma * fy(x, y))
            x, y = x_new.copy(), y_new.copy()
            snakes.append((x.copy(), y.copy()))

        return snakes


class ImageProcessor:
    def __init__(self, img_path, x_center, y_center, rx, ry, n_points, contour_params):
        self.img_path = img_path
        self.x_center = x_center
        self.y_center = y_center
        self.rx = rx
        self.ry = ry
        self.n_points = n_points
        self.contour_params = contour_params

    def load_image(self):
        original_img = mimg.imread(self.img_path)
        if len(original_img.shape) == 3:
            gray_img = np.mean(original_img, axis=2)
        else:
            gray_img = original_img.copy()
        return original_img, gray_img

    def initialize_contour(self):
        t = np.linspace(0, 2*np.pi, self.n_points, endpoint=False)
        x = self.x_center + self.rx * np.cos(t)
        y = self.y_center + self.ry * np.sin(t)
        return x, y

    def process(self):
        original_img, gray_img = self.load_image()
        x, y = self.initialize_contour()
        contour = ActiveContour(**self.contour_params)
        fx, fy = contour.create_external_edge_force_gradients(gray_img)
        snakes = contour.iterate(x, y, fx, fy)

        # Visualization
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
        param_text = f"Parameters:\nα={self.contour_params['alpha']}\nβ={self.contour_params['beta']}\nγ={self.contour_params['gamma']}\nσ={self.contour_params['sigma']}"
        ax.text(0.98, 0.98, param_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.8))

        ax.legend()
        plt.title(f"Active Contour on {self.img_path}")
        plt.show()


if __name__ == "__main__":
    # Define contour parameters for each image
    trefle_params = {'alpha': 0.03, 'beta': 0.1, 'gamma': 10, 'sigma': 2.0, 'n_iters': 10000}
    pique_params = {'alpha': 0.03, 'beta': 0.001, 'gamma': 2, 'sigma': 1.0, 'n_iters': 16000}
    etoile_params = {'alpha': 0.01, 'beta': 0.001, 'gamma': 8, 'sigma': 1.0, 'n_iters': 16000}
    vase_params = {'alpha': 0.001, 'beta': 1, 'gamma': 30, 'sigma': 1.0, 'n_iters': 20000}

    # Process each image
    print("Processing trèfle image...")
    trefle_processor = ImageProcessor('img/trefle.jpg', 150, 150, 150, 150, 200, trefle_params)
    trefle_processor.process()

    print("Processing pique image...")
    pique_processor = ImageProcessor('img/pique.jpg', 550, 550, 450, 450, 300, pique_params)
    pique_processor.process()

    print("Processing etoile image...")
    etoile_processor = ImageProcessor('img/etoile.png', 320, 320, 340, 340, 200, etoile_params)
    etoile_processor.process()

    print("Processing vase image...")
    vase_processor = ImageProcessor('img/vase.jpg', 100, 90, 70, 70, 200, vase_params)
    vase_processor.process()