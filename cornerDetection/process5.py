import cv2
import numpy as np  # Agregar esta línea

cv2.destroyAllWindows()


# Función para manejar eventos de clic del mouse
def mouse_event(event, x, y, flags, param):
    global corners, cropping, cropped

    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append((x, y))
        cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Selecciona las esquinas', img_copy)

        if len(corners) == 4:
            cropping = True

# Cargar la imagen
img = cv2.imread('./scan_02.jpg')

# Crear una copia de la imagen
img_copy = img.copy()

# Inicializar variables
corners = []
cropping = False
cropped = False

# Crear una ventana para mostrar la imagen y configurar el manejador de eventos
cv2.namedWindow('Selecciona las esquinas')
cv2.setMouseCallback('Selecciona las esquinas', mouse_event)

while True:
    cv2.imshow('Selecciona las esquinas', img_copy)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if cropping:
        # Ordenar las esquinas en sentido horario
        corners = sorted(corners, key=lambda corner: (corner[0], corner[1]))

        # Recortar la región seleccionada
        x, y, w, h = cv2.boundingRect(np.array(corners))
        cropped_invoice = img[y:y + h, x:x + w]

        # Mostrar la factura recortada
        cv2.imshow('Factura Recortada', cropped_invoice)
        cv2.waitKey(0)
        cropped = True


cv2.destroyAllWindows()
   



