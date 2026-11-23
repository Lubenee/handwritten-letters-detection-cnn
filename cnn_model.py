from keras import layers
import tensorflow as tf
from latin_letter_dataset import LETTERS

def create_cnn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28,28,1)), 
        tf.keras.layers.Conv2D(128, (3,3), 1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Conv2D(256, (3,3), 1, activation='sigmoid'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='sigmoid'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(LETTERS), activation='softmax'),
    ])