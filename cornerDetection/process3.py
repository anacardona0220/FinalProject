import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('./scan_02.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar el detector de esquinas Harris
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Aumentar el contraste de las esquinas
corners = cv2.dilate(corners, None)

# Definir un umbral para la detección de esquinas
threshold = 0.01 * corners.max()

# Encontrar las coordenadas de las esquinas
corner_coordinates = np.argwhere(corners > threshold)

# Dibujar círculos rojos en las esquinas
for corner in corner_coordinates:
    x, y = corner[1], corner[0]
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Mostrar la imagen con las esquinas detectadas
cv2.imshow('Esquinas detectadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
