import cv2

# Cargar la imagen
image = cv2.imread('./scan_02.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de borde para resaltar los contornos
edges = cv2.Canny(gray, threshold1=30, threshold2=100)

# Encontrar contornos en la imagen
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encuentra el contorno más grande (que debería ser la factura)
largest_contour = max(contours, key=cv2.contourArea)

# Aproximar el contorno con un polígono
epsilon = 0.04 * cv2.arcLength(largest_contour, True)
approx = cv2.approxPolyDP(largest_contour, epsilon, True)

# Dibujar el polígono en la imagen original
cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# Mostrar la imagen con los contornos dibujados
cv2.imshow('Contornos de la factura', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
