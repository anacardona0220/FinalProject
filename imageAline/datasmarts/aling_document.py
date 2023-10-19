# Importamos las dependencias del programa.
from argparse import ArgumentParser

import cv2
from imutils import resize
import numpy as np

def align_images(image, template, max_features=500, keep_percent=0.2, debug=False):
    """
    Esta función alínea una imagen con respecto a una plantilla de referencia.
    :param image: Imagen a ser alineada.
    :param template: Imagen plantilla usada como referencia para la alineación.
    :param max_features: Cota superior para el número de regiones candidatas a usar para la alineación.
    :param keep_percent: Porcentaje de keypoints a mantener.
    :param debug:
    :return:
    """

        # Convertimos las imágenes a escala de grises.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Creamos el descriptor ORB.
    orb = cv2.ORB_create(max_features)

    # Extraemos los puntos clave y los descriptores de cada imagen.
    keypoints_image, descriptors_image = orb.detectAndCompute(image_gray, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(template_gray, None)

        # Emparejamos los descriptores de la imagen y de la plantilla.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors_image, descriptors_template, None)

    # Ordenamos las parejas con base a su cercanía.
    matches = sorted(matches, key=lambda x: x.distance)

        # Nos quedamos con un porcentaje de las mejores parejas.
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    if debug:
        # Creamos una visualización de los puntos claves emparejados entre la imagen y la plantilla.
        matched_visualization = cv2.drawMatches(image, keypoints_image, template, keypoints_template, matches, None)
        matched_visualization = resize(matched_visualization, width=1000)
        cv2.imshow('Puntos claves emparejados', matched_visualization)
        cv2.waitKey(0)

    points_image = np.zeros((len(matches), 2), dtype='float')
    points_template = np.zeros((len(matches), 2), dtype='float')

    # Iteramos sobre los matches para mapear los puntos clave de la imagen con los de la plantilla.
    for i, m in enumerate(matches):
        points_image[i] = keypoints_image[m.queryIdx].pt
        points_template[i] = keypoints_template[m.trainIdx].pt

            # Hallamos la homografía entre las dos colecciones de puntos.
    # Dicha homografía es una matriz que nos permitirá alterar la perspectiva de la imagen original, para que coincida
    # con la de la plantilla (o, en resumidas cuentas, alinear la imagen con la plantilla).
    H, mask = cv2.findHomography(points_image, points_template, method=cv2.RANSAC)

        # Alineamos y retornamos la imagen usando la homografía calculada anteriormente.
    h, w = template.shape[:2]
    return cv2.warpPerspective(image, H, (w, h))

# Definimos los argumentos de entrada del programa.
argument_parser = ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True,
                             help='Ruta a la imagen de entrada que alinearemos con respecto al template.')
argument_parser.add_argument('-t', '--template', required=True, help='Ruta a la imagen de entrada del template.')
arguments = vars(argument_parser.parse_args())

# Alineamos la imagen con relación a la plantilla.
aligned = align_images(image, template, debug=True)

# Redimensionamos la imagen alineada y la plantilla.
aligned = resize(aligned, width=700)
template = resize(template, width=700)

# Juntamos las dos imágenes para mostrarlas como una sola.
stacked = np.hstack([aligned, template])

# Montaremos una imagen sobre la otra para validar visualmente la alineación.
overlay = template.copy()
output = aligned.copy()
cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

# Mostramos el resultado, primero individualmente, y luego de forma superpuesta.
cv2.imshow('IMAGENES ALINEADAS', stacked)
cv2.imshow('IMAGENES SUPERPUESTAS', output)
cv2.waitKey(0)

# Destruimos las ventanas creadas durante la ejecución del programa.
cv2.destroyAllWindows()

