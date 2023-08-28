# Model -3 Transfer Learning with USE Encoder
print("RUNNING Model -3 Transfer Learning with USE Encoder")

import tensorflow_hub as hub
from tensorflow import keras
import helper
from tensorflow.keras import layers
import tensorflow as tf



def build():
    # model with Sequential api
    model_3 = keras.Sequential()

    # universal-sentence-encoder layer
    # directly from tfhub
    use_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                               trainable=False,
                               input_shape=[],
                               dtype=tf.string,
                               name='USE')
    model_3.add(use_layer)
    model_3.add(layers.Dropout(0.2))
    model_3.add(layers.Dense(64, activation=keras.activations.relu))
    model_3.add(layers.Dense(1, activation=keras.activations.sigmoid))

    helper.compile_model(model_3)

    return model_3
