import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread('./scan_02.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de Canny para detectar bordes
edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# Encontrar las líneas en la imagen
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Encontrar las cuatro líneas más largas que forman un cuadrilátero
# Encontrar las cuatro líneas más largas que forman un cuadrilátero
lines = [line[0] for line in lines]
lines = sorted(lines, key=lambda line: line[0], reverse=True)[:4]



# Extraer las coordenadas de las esquinas de las cuatro líneas
corners = []
for rho, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    corners.append((x1, y1))
    corners.append((x2, y2))

# Ordenar las esquinas en sentido horario o antihorario
center = np.mean(corners, axis=0)
corners = sorted(corners, key=lambda corner: np.arctan2(corner[1] - center[1], corner[0] - center[0]))

# Recortar la región de la factura
x, y, w, h = cv2.boundingRect(np.array(corners))
cropped_invoice = image[y:y+h, x:x+w]

# Mostrar la factura recortada
cv2.imshow('Factura Recortada', cropped_invoice)
cv2.waitKey(0)
cv2.destroyAllWindows()
