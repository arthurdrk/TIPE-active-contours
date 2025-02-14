from main import *
from PIL import Image

# Ouvrir une image
image = Image.open('./img/trefle.jpg')

rectangle(image, 400, 400)
evolution_contour(image, 100, 100, 0.1, 1, 0.1, 100, 10, 10, 10)
