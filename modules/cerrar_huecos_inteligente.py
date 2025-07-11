import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import reconstruction
from skimage.morphology import disk
from scipy.ndimage import binary_erosion

def cerrar_huecos_inteligente(mascara_binaria, erosion_iter=2, mostrar=True):
    """
    Cierra huecos internos parcialmente abiertos sin unir células completas.

    :param mascara_binaria: np.ndarray binaria 0-255
    :param erosion_iter: cantidad de erosiones para separar bien los huecos
    :param mostrar: mostrar comparación visual
    :return: máscara con huecos cerrados (np.uint8)
    """
    # Asegurar que sea 0-1
    binaria = (mascara_binaria > 0).astype(np.uint8)

    # Invertimos: huecos ahora son 1, fondo 0
    invertida = 1 - binaria

    # Marcadores: erosión → evita bordes y se queda con huecos internos
    marcadores = binary_erosion(invertida, structure=np.ones((3, 3)), iterations=erosion_iter)

    # Reconstrucción morfológica: solo expande desde marcadores
    reconstruida = reconstruction(
        seed=marcadores.astype(np.uint8),
        mask=invertida,
        method='dilation'
    )

    # Volver a binaria normal (cerramos huecos)
    cerrado = 1 - reconstruida.astype(np.uint8)
    cerrado = cerrado * 255  # 0-255

    if mostrar:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mascara_binaria, cmap='gray')
        plt.title("Original (con huecos)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cerrado, cmap='gray')
        plt.title("Huecos cerrados (reconstrucción)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return cerrado.astype(np.uint8)
