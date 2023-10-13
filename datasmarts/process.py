# Importamos las dependencias del proyecto.
from argparse import ArgumentParser

import cv2
import imutils
import numpy as np
import pytesseract

# Definimos los argumentos de entrada del programa.
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--image', type=str, required=True, help='Ruta a la imagen de entrada.')
arguments = vars(argument_parser.parse_args())

# Cargamos la imagen, y la convertimos a escala de grises.
image = cv2.imread(arguments['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicamos thresholding automático con el algoritmo de Otsu. ESto hará que el texto se vea blanco, y los elementos
# del fondo sean menos prominentes.
thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Mostramos el resultado parcial en pantalla
cv2.imshow('Otsu', thresholded)
cv2.waitKey(0)

# Calculamos y normalizamos la transformada de distancia.
dist = cv2.distanceTransform(thresholded, cv2.DIST_L2, 5)
dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
dist = (dist * 255).astype('uint8')

# Mostramos el resultado parcial en pantalla.
cv2.imshow('Dist', dist)
cv2.waitKey(0)

# Aplicamos thresholding al resultado de la operación anterior, y mostramos el resultado en pantalla.
dist = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('Dist Otsu', dist)
cv2.waitKey(0)

# Aplicamos apertura para desconectar manchas y blobs de los elementos que nos interesan (los números)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
opening = cv2.morphologyEx(dist, cv2.MORPH_OPEN, kernel)
cv2.imshow('Apertura', opening)
cv2.waitKey(0)

# Hallamos los contornos de los números en la imagen.
contours = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

chars = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Solo los contornos grandes perdurarán, ya que corresponden a los números que nos interesan.
    if w >= 35 and h >= 100:
        chars.append(contour)

# Hallamos la cáscara convexa que envuelve todos los números.
chars = np.vstack([chars[i] for i in range(0, len(chars))])
hull = cv2.convexHull(chars)

# Creamos una máscara y la alargamos.
mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.drawContours(mask, [hull], -1, 255, -1)
mask = cv2.dilate(mask, None, iterations=2)
cv2.imshow('MASCARA', mask)

# Aplicamos la máscara para aislar los números del fondo.
final = cv2.bitwise_and(opening, opening, mask=mask)

# Extraemos los dígitos de la imagen. Nota el uso de un whitelist de sólo números.
options = '-c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(final, config=options)
print(text)

# Mostramos la imagen resultante de todo el procesamiento.
cv2.imshow('Final', final)
cv2.waitKey(0)

# Destruimos las ventanas creadas durante la ejecución del programa.
cv2.destroyAllWindows()


# Mostramos la imagen resultante de todo el procesamiento.
cv2.imshow('Final', final)
cv2.waitKey(0)

# Guardar la imagen procesada en un archivo
cv2.imwrite('./challenge.png', final)


# Destruimos las ventanas creadas durante la ejecución del programa.
cv2.destroyAllWindows()
