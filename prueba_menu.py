import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

# ESTA YA ES LA PARTE PARA PODER VER LA PARTE DEL MENU PARA LOS SISTEMAS O OPCIONES xd

img_racoon_bgr = cv2.imread("racoon_city_XD.png", cv2.IMREAD_COLOR)
img_racoon_rgb = img_racoon_bgr[:, :, ::-1]

cv2.imwrite("racoon_country.png", img_racoon_rgb)

plt.imshow(img_racoon_rgb)