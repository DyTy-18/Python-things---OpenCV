import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

img_racoon_bgr = cv2.imread("racoon.png", cv2.IMREAD_COLOR)

# -----------------------------------------------------

""" cv2.imwrite("racoon_destino.png", img_racoon_bgr)
cv2.imshow("Imagen Original", img_racoon_bgr)
cv2.waitKey(0)  
cv2.destroyAllWindows() """

# -----------------------------------------------------

""" if img_racoon_bgr is not None:
    alto, ancho = img_racoon_bgr.shape[:2]
    canales = img_racoon_bgr.shape[2] if len(img_racoon_bgr.shape) == 3 else 1
    
    print("Alto de la imagen:", alto)
    print("Ancho de la imagen:", ancho)
    print("Número de canales de la imagen:", canales)
    plt.imshow(img_racoon_bgr)
    cv2.imshow("Imagen Original", img_racoon_bgr)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()
    
else:
    print("No se pudo cargar la imagen.") """

# -----------------------------------------------------

""" if img_racoon_bgr is not None:
    # Convertir la imagen a blanco y negro
    img_racoon_gris = cv2.cvtColor(img_racoon_bgr, cv2.COLOR_BGR2GRAY)

    alto, ancho = img_racoon_bgr.shape[:2]
    
    for y in range(alto):
        for x in range(ancho):
            pixel = img_racoon_gris[y, x]   
            if pixel == 0:  # Si el pixel es negro en blanco y negro
                color_aleatorio = np.random.randint(0, 256) 
                img_racoon_bgr[y, x] = [color_aleatorio, color_aleatorio, color_aleatorio]

    # Guardar la imagen modificada
    cv2.imwrite("racoon_destino_colores_aleatorios_bn.png", img_racoon_bgr)

    # Mostrar la imagen modificada
    plt.imshow(cv2.cvtColor(img_racoon_bgr, cv2.COLOR_BGR2RGB))
    plt.show()
else:
    print("No se pudo cargar la imagen.") """

# -----------------------------------------------------

""" if img_racoon_bgr is not None:
    img_girada_90 = cv2.rotate(img_racoon_bgr, cv2.ROTATE_90_CLOCKWISE)
    
    cv2.imwrite("racoon_girada_90.png", img_girada_90)
    cv2.imshow("Imagen Modificada", img_girada_90)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    plt.imshow(cv2.cvtColor(img_girada_90, cv2.COLOR_BGR2RGB))
    plt.imshow(img_girada_90)

else:
    print("No se pudo cargar la imagen.") """

# ---------------------------------------------------------

""" if img_racoon_bgr is not None:
    x_inicio, x_fin = 300, 600
    y_inicio, y_fin = 200, 400 

    # Recortar la región de interés (ROI) de la imagen
    roi_recortada = img_racoon_bgr[y_inicio:y_fin, x_inicio:x_fin]

    # Guardar la región de interés (ROI) recortada
    cv2.imwrite("racoon_recortada.png", roi_recortada)

    cv2.imshow("Imagen Modificada", roi_recortada)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    plt.imshow(roi_recortada)

else:
    print("No se pudo cargar la imagen.") """

# -----------------------------------------------------

""" if img_racoon_bgr is not None:
    # Definir el texto que deseas escribir
    texto = "La respuesta de la vida es 42"

    coordenadas_inicio = (50, 50)

    fuente = cv2.FONT_HERSHEY_SIMPLEX
    tamaño_fuente = 1
    color_texto = (255, 0, 0) 

    img_con_texto = cv2.putText(img_racoon_bgr, texto, coordenadas_inicio, fuente, tamaño_fuente, color_texto, thickness=2)

    cv2.imwrite("racoon_con_texto.png", img_con_texto)

    cv2.imshow("Imagen Modificada", img_con_texto)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    plt.imshow(img_con_texto)

else:
    print("No se pudo cargar la imagen.")
 """
# -----------------------------------------------------

""" if img_racoon_bgr is not None:
    
    # Dibujar un rectángulo
    cv2.rectangle(img_racoon_bgr, (50, 50), (200, 200), (0, 255, 0), thickness=2)  

    # Dibujar un círculo
    cv2.circle(img_racoon_bgr, (300, 150), 100, (0, 0, 255), thickness=2)  

    # Dibujar una línea
    cv2.line(img_racoon_bgr, (400, 50), (500, 200), (255, 0, 0), thickness=2)  

    cv2.imwrite("racoon_con_formas.png", img_racoon_bgr)

    cv2.imshow("Imagen Modificada", img_racoon_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    plt.imshow(img_racoon_bgr)

else:
    print("No se pudo cargar la imagen.")
 """
# -----------------------------------------------------

captura_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break
    # Escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en el frame
    rostros = captura_rostro.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un recuadro alrededor de los rostros detectados
    for (x, y, w, h) in rostros:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)

    # Mostrar el frame con los recuadros alrededor de los rostros
    cv2.imshow("Detección de Rostros", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()