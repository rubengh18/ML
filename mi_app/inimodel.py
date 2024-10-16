import tensorflow as tf

import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_modelo = os.path.join(BASE_DIR, 'my_h5_model.h5')

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))   
mobilenetv2.trainable = False                 
modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(3, activation='softmax')
])
modelo.summary()                         
modelo.load_weights(ruta_modelo)  # Actualiza con la ruta correcta de tu modelo

# Exportar el modelo para ser utilizado en otras partes del proyecto
model = modelo
