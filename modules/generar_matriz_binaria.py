import numpy as np
import matplotlib.pyplot as plt  # Importamos Matplotlib para visualizar
from skimage import io
import numpy as np

def generar_matriz_binaria(imagen_umbralizada):
    """
    Crea una matriz binaria a partir de una imagen (RGB o gris) umbralizada.
    
    Parámetros:
        - imagen_umbralizada (np.ndarray): Imagen umbralizada (grises o RGB).
    
    Retorna:
        - np.ndarray: Matriz binaria de 0 (negro) y 1 (resto de los colores).
    """
    # 📌 Si es RGB, sumar los canales. Si es gris, usarla directamente.
    if len(imagen_umbralizada.shape) == 3:
        imagen_gris = np.sum(imagen_umbralizada, axis=2)
    else:
        imagen_gris = imagen_umbralizada

    # 📌 Mostrar para depuración
    plt.imshow(imagen_gris, cmap='gray')
    plt.axis('off')
    plt.title("Imagen en Gris para Máscara")
    plt.show()

    # 📌 Crear matriz binaria: 1 si el píxel no es negro, 0 si es negro
    matriz_binaria = (imagen_gris > 0).astype(np.uint8)

    return matriz_binaria

