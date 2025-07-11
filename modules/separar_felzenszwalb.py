from skimage.segmentation import felzenszwalb
from skimage.color import gray2rgb
import matplotlib.pyplot as plt 

def separar_felzenszwalb(imagen_gris, scale=100, sigma=0.5, min_size=50, mostrar=True):
    """
    Aplica segmentación con el algoritmo de Felzenszwalb.

    :param imagen_gris: Imagen en escala de grises.
    :return: Etiquetas por región.
    """
    imagen_color = gray2rgb(imagen_gris)
    etiquetas = felzenszwalb(imagen_color, scale=scale, sigma=sigma, min_size=min_size)

    if mostrar:
        plt.figure(figsize=(6, 6))
        plt.imshow(etiquetas, cmap='tab20')
        plt.title("Segmentación - Felzenszwalb")
        plt.axis('off')
        plt.show()

    return etiquetas
