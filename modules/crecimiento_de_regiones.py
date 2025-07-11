import numpy as np
import cv2
import matplotlib.pyplot as plt

def crecimiento_de_regiones(imagen_gris, semilla, tolerancia=5, mostrar=True):
    """
    Realiza crecimiento de regiones desde un punto semilla en una imagen en escala de grises.
    
    :param imagen_gris: Imagen en escala de grises (np.ndarray, dtype=uint8 o float32 normalizada).
    :param semilla: Tupla (fila, columna) con la posición inicial.
    :param tolerancia: Diferencia máxima de intensidad permitida con la semilla.
    :param mostrar: Si True, muestra la máscara obtenida.
    :return: Máscara binaria (uint8) con la región crecida.
    """
    if imagen_gris.dtype != np.uint8:
        imagen_gris = (imagen_gris * 255).astype(np.uint8)

    altura, ancho = imagen_gris.shape
    visitado = np.zeros((altura, ancho), dtype=bool)
    mascara = np.zeros_like(imagen_gris, dtype=np.uint8)

    valor_semilla = imagen_gris[semilla]
    cola = [semilla]

    while cola:
        y, x = cola.pop()
        if visitado[y, x]:
            continue
        visitado[y, x] = True

        if abs(int(imagen_gris[y, x]) - int(valor_semilla)) <= tolerancia:
            mascara[y, x] = 255

            # Añadir vecinos válidos
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < altura and 0 <= nx < ancho and not visitado[ny, nx]:
                        cola.append((ny, nx))

    if mostrar:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(imagen_gris, cmap='gray')
        plt.plot(semilla[1], semilla[0], 'ro')
        plt.title("Imagen + semilla")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mascara, cmap='gray')
        plt.title("Resultado crecimiento")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return mascara
