import os
import numpy as np
import tensorflow as tf
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- 1. Parámetros ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
BATCH_SIZE = 8
EPOCHS = 25

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("mejor_modelo.h5", save_best_only=True) 
]


# --- 2. Carga de rutas ---
image_dir = "data/images/"
mask_dir = "data/masks/"

image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])

# --- 3. Dataset Generator ---
class DataGenerator(Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, img_size):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for img_path, mask_path in zip(batch_x, batch_y):
            img = load_img(img_path, target_size=self.img_size)
            img = img_to_array(img) / 255.0

            mask = load_img(mask_path, color_mode="grayscale", target_size=self.img_size)
            mask = img_to_array(mask) / 255.0
            mask = (mask > 0.5).astype(np.float32)  # binarizar

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

# --- 4. División del dataset ---
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

train_gen = DataGenerator(train_images, train_masks, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH))
val_gen = DataGenerator(val_images, val_masks, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH))

# --- 5. Modelo U-Net ---
def unet_model(input_size=(128, 128, 3), num_classes=1):
    inputs = tf.keras.Input(input_size)

    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)

    u6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# --- 6. Compilar y entrenar ---
model = unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', iou_metric]
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- 7. Guardar modelo ---
model.save("unet_segmentacion.keras")

# Graficar
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.plot(history.history['iou_metric'], label='IoU entrenamiento')
plt.plot(history.history['val_iou_metric'], label='IoU validación')
plt.legend()
plt.title("Curvas de entrenamiento")
plt.show()

""" Agregar el indice DICE
Agregar la matriz de confusión
Deberia controlar que no se sobreentrene
Deberia ser capaz de indentificar la cantidad de epocos optimas
Me deberia guardar el mejor conjunto de pesos, no que se quede con el ultimo 
BATCH_SIZE se debe ajustar a la cantidad de imagenes del dataset 

deberia cambiar la division de datos 70% para entrar 30% para probar:
train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)
 """