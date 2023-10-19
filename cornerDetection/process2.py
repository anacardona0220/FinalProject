import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('./scan_02.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de borde para resaltar los contornos
edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# Encontrar contornos en la imagen
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrar el contorno más grande (que debería ser la factura)
largest_contour = max(contours, key=cv2.contourArea)

# Encontrar las esquinas del contorno
epsilon = 0.04 * cv2.arcLength(largest_contour, True)
corners = cv2.approxPolyDP(largest_contour, epsilon, True)

# Dibujar círculos en las esquinas
for corner in corners:
    x, y = corner[0]
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Mostrar la imagen con las esquinas detectadas
cv2.imshow('Esquinas detectadas', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
