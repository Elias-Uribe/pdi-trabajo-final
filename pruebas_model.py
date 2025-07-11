import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2

# -----------------------------
# 1. Cargar modelo U-Net entrenado
# -----------------------------
modelo = tf.keras.models.load_model("unet_segmentacion.keras", compile=False)

# -----------------------------
# 2. Ruta de imagen de prueba
# -----------------------------
ruta_imagen = "Tareas/Proyecto final - Elias Uribe/images/image-83.png"  # Cambiá esto por la imagen que quieras probar

# -----------------------------
# 3. Preprocesamiento
# -----------------------------
IMG_SIZE = (256, 256)  # Tamaño usado durante el entrenamiento

# Cargar imagen y convertir a array normalizado
img = load_img(ruta_imagen, target_size=IMG_SIZE)
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión para batch

# -----------------------------
# 4. Predicción
# -----------------------------
pred = modelo.predict(img_array)[0, :, :, 0]  # Remover batch y canal extra

# Binarizar máscara
mascara_binaria = (pred > 0.5).astype(np.uint8) * 255

# -----------------------------
# 5. Mostrar resultados
# -----------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("🩸 Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mascara_binaria, cmap="gray")
plt.title("🧠 Máscara segmentada (U-Net)")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------
# 6. Guardar la máscara
# -----------------------------
""" nombre_salida = "mascara_generada.png"
cv2.imwrite(nombre_salida, mascara_binaria)
print(f"✅ Máscara guardada como {nombre_salida}") """
