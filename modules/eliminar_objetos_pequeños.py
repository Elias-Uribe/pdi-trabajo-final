import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects

def eliminar_objetos_pequeños(mascara_binaria, min_pixeles=50, mostrar=True):
    """
    Elimina objetos pequeños (ruido) de una máscara binaria.
    
    :param mascara_binaria: Imagen binaria (np.uint8 o bool), donde los objetos son blancos.
    :param min_pixeles: Tamaño mínimo en píxeles para conservar un objeto.
    :param mostrar: Si True, muestra comparación antes/después.
    :return: Máscara binaria limpia (np.uint8).
    """
    # Asegurarse de que sea booleano para usar remove_small_objects
    mascara_bool = (mascara_binaria > 0)

    # Remover objetos pequeños
    limpia_bool = remove_small_objects(mascara_bool, min_size=min_pixeles)

    # Convertir a uint8 para seguir trabajando
    limpia = (limpia_bool * 255).astype(np.uint8)

    if mostrar:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mascara_binaria, cmap='gray')
        plt.title("Antes (original)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(limpia, cmap='gray')
        plt.title(f"Después (eliminado < {min_pixeles} px)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return limpia