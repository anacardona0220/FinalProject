import cv2

# Cargar la imagen
image = cv2.imread('./scan_02.jpg')

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar esquinas con el algoritmo Shi-Tomasi
corners = cv2.goodFeaturesToTrack(gray, maxCorners=4, qualityLevel=0.01, minDistance=10)
corners = corners.astype(int)

# Obtener las coordenadas de las esquinas
x, y = corners[:, 0, 0], corners[:, 0, 1]

# Calcular los límites del rectángulo que rodea las esquinas
x_min, x_max, y_min, y_max = min(x), max(x), min(y), max(y)

# Recortar la región de la factura
cropped_invoice = image[y_min:y_max, x_min:x_max]

# Guardar la región recortada en un nuevo archivo
cv2.imwrite('factura_recortada.jpg', cropped_invoice)
