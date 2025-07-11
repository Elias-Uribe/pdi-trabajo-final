import numpy as np

def datos_estadisticos(imagen, mascara=None, mostrar=True):
    """
    Calcula la media y el desvío estándar de una imagen (o dentro de una máscara).
    
    :param imagen: ndarray, imagen 2D en escala de grises.
    :param mascara: ndarray opcional (mismo tamaño) para calcular sobre una región específica (valores 0 y 255 o bool).
    :param mostrar: Si True, imprime los resultados.
    :return: media, std
    """
    if mascara is not None:
        # Asegurar que la máscara esté en formato booleano
        mascara = mascara > 0
        valores = imagen[mascara]
    else:
        valores = imagen.flatten()
    
    media = np.mean(valores)
    std = np.std(valores)

    if mostrar:
        print(f"📊 Estadísticas:")
        print(f"  ▸ Media: {media:.2f}")
        print(f"  ▸ Desvío estándar: {std:.2f}")
    
    return media, std