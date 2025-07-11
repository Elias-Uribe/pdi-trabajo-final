from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import numpy as np

def cerrar_huecos_precisos(mascara_binaria, mostrar=True):
    """
    Cierra únicamente los huecos internos de los glóbulos sin alterar bordes externos.
    """
    mascara_filled = binary_fill_holes(mascara_binaria > 0).astype(np.uint8) * 255

    if mostrar:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mascara_binaria, cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mascara_filled, cmap='gray')
        plt.title("Huecos internos cerrados")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return mascara_filled
