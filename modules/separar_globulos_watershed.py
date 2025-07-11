import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

def separar_globulos_watershed(mascara_binaria, min_pixeles=100, min_distance=10, mostrar=True):
 
    """ Mejora la separación de glóbulos pegados usando watershed con preprocesamiento.
    
    :param mascara_binaria: Imagen binaria (0 y 255) con los objetos blancos.
    :param min_pixeles: Tamaño mínimo de objeto para conservar.
    :param min_distance: Distancia mínima entre picos (controla separación de objetos).
    :param mostrar: Si True, muestra los pasos visualmente.
    :return: Etiquetas separadas (cada objeto tiene su número distinto).
     """
    # 1. Preprocesamiento
    # Convertir a bool, cerrar huecos internos y limpiar
    mascara_bool = binary_fill_holes(mascara_binaria > 0)
    mascara_bool = remove_small_objects(mascara_bool, min_size=min_pixeles)

    # 2. Suavizado leve (erosión leve para "afilar" centros)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mascara_suavizada = cv2.erode(mascara_bool.astype(np.uint8) * 255, kernel, iterations=1)

    # 3. Mapa de distancia
    distancia = ndi.distance_transform_edt(mascara_suavizada)

    # 4. Detectar máximos locales (centros de glóbulos)
    coords_maximos = peak_local_max(
        distancia,
        min_distance=min_distance,
        threshold_abs=distancia.max() * 0.4,
        labels=mascara_suavizada
    )

    # 5. Crear marcadores
    marcadores = np.zeros_like(mascara_suavizada, dtype=np.int32)
    for i, coord in enumerate(coords_maximos):
        marcadores[coord[0], coord[1]] = i + 1

    # 6. Aplicar watershed
    etiquetas = watershed(-distancia, markers=marcadores, mask=mascara_bool)

    # 7. Mostrar resultados
    if mostrar:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(mascara_binaria, cmap='gray')
        plt.title("Máscara original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(distancia, cmap='jet')
        plt.title("Mapa de distancia")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(etiquetas, cmap='tab20')
        plt.title("Separación (watershed)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return etiquetas
