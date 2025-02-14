from main import *
from PIL import Image

# Ouvrir une image
image = np.array(Image.open('./img/trefle.jpg'))

X,Y=ellipse(centre=(150,150),rx=140,ry=140,n=100)
evolution_contour(image, X, Y, alpha=0.001, beta=0.001, gamma=0.1, n_iters=2000, lambd=0.3, Fnorm=1, sigma=0.3)
