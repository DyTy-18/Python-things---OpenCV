import cv2
import numpy as np
import matplotlib.pyplot as plt

def cargar_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    return img

def mostrar_propiedades(img):
    if img is not None:
        alto, ancho = img.shape[:2]
        canales = img.shape[2] if len(img.shape) == 3 else 1
        
        print("Alto de la imagen:", alto)
        print("Ancho de la imagen:", ancho)
        print("Número de canales de la imagen:", canales)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        cv2.imshow("Imagen Original", img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()
    else:
        print("No se pudo cargar la imagen.")

def girar_imagen(img):
    if img is not None:
        img_girada_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("Imagen Girada 90°", img_girada_90)
        cv2.imwrite("racoon_girada_90.png", img_girada_90)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        plt.imshow(cv2.cvtColor(img_girada_90, cv2.COLOR_BGR2RGB))
    else:
        print("No se pudo cargar la imagen.")

def recortar_imagen(img):
    if img is not None:
        x_inicio, x_fin = 300, 600
        y_inicio, y_fin = 200, 400 
        roi_recortada = img[y_inicio:y_fin, x_inicio:x_fin]
        cv2.imshow("Imagen Recortada", roi_recortada)
        cv2.imwrite("racoon_recortada.png", roi_recortada)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        plt.imshow(roi_recortada)
    else:
        print("No se pudo cargar la imagen.")

def agregar_texto(img):
    if img is not None:
        texto = "La respuesta de la vida es 42"
        coordenadas_inicio = (50, 50)
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        tamaño_fuente = 1
        color_texto = (255, 0, 0) 
        img_con_texto = cv2.putText(img, texto, coordenadas_inicio, fuente, tamaño_fuente, color_texto, thickness=2)
        cv2.imwrite("racoon_con_texto.png", img_con_texto)
        cv2.imshow("Imagen con Texto", img_con_texto)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        plt.imshow(img_con_texto)
    else:
        print("No se pudo cargar la imagen.")

def dibujar_formas(img):
    if img is not None:
        cv2.rectangle(img, (50, 50), (200, 200), (0, 255, 0), thickness=2)  
        cv2.circle(img, (300, 150), 100, (0, 0, 255), thickness=2)  
        cv2.line(img, (400, 50), (500, 200), (255, 0, 0), thickness=2)  
        cv2.imwrite("racoon_con_formas.png", img)
        cv2.imshow("Imagen con Formas", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        plt.imshow(img)
    else:
        print("No se pudo cargar la imagen.")

def detectar_rostros_camara():
    captura_rostro = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("No se pudo abrir la cámara.")
        return
    while True:
        ret, frame = camara.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostros = captura_rostro.detectMultiScale(frame_gris, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in rostros:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
        cv2.imshow("Detección de Rostros", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camara.release()
    cv2.destroyAllWindows()

def menu():
    print("Menu de opciones:")
    print("1. Cargar imagen y mostrar propiedades")
    print("2. Girar imagen 90 grados")
    print("3. Recortar imagen")
    print("4. Agregar texto a la imagen")
    print("5. Dibujar formas en la imagen")
    print("6. Detectar rostros con la cámara")
    print("0. Salir")

def main():
    while True:
        menu()
        opcion = input("Ingrese el número de la opción deseada: ")
        if opcion == "0":
            print("Saliendo del programa...")
            break
        elif opcion == "1":
            ruta = input("Ingrese la ruta de la imagen: ")
            img = cargar_imagen(ruta)
            mostrar_propiedades(img)
        elif opcion == "2":
            img = cargar_imagen("racoon.png")
            girar_imagen(img)
        elif opcion == "3":
            img = cargar_imagen("racoon.png")
            recortar_imagen(img)
        elif opcion == "4":
            img = cargar_imagen("racoon.png")
            agregar_texto(img)
        elif opcion == "5":
            img = cargar_imagen("racoon.png")
            dibujar_formas(img)
        elif opcion == "6":
            detectar_rostros_camara()
        else:
            print("Opción no válida. Por favor, seleccione una opción válida.")

if __name__ == "__main__":
    main()
