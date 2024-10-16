import tensorflow as tf

def cargar_y_redimensionar_imagen(ruta_imagen):
    """
    Carga una imagen en un tensor de TensorFlow y la redimensiona a 224x224x3.

    Args:
      ruta_imagen: Ruta de la imagen que se va a cargar.

    Devoluciones:
      Un tensor de TensorFlow que contiene la imagen redimensionada.
    """
    # Lee la imagen en modo de color RGB
    imagen = tf.io.read_file(ruta_imagen)
    imagen = tf.image.decode_jpeg(imagen, channels=3)

    # Redimensiona la imagen a 224x224
    imagen_redimensionada = tf.image.resize(imagen, [224, 224])

    # Expande la dimensión para que coincida con la forma (224, 224, 3)
    imagen_redimensionada = tf.expand_dims(imagen_redimensionada, 0)

    return imagen_redimensionada

def make_output(logits):
    out_labels = ['melanoma_a', 'melanoma_i', 'no_melanoma']
    output = tf.argmax(logits, axis=1)
    return out_labels[output[0]]
