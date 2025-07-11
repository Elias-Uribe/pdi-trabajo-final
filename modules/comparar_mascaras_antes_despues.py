import matplotlib.pyplot as plt

def comparar_mascaras_antes_despues(original, procesada, titulo="Separación por apertura"):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Antes")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(procesada, cmap='gray')
    plt.title("Después")
    plt.axis("off")

    plt.suptitle(titulo)
    plt.tight_layout()
    plt.show()
