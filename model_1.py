# Model 1: Creating custom Text vectorization and embedding layers:
print("RUNNING Model 1: Creating custom Text vectorization and embedding layers")
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras


def build(text_vec, embedding_layer):

    # Now, let’s build and compile model 1 using the Tensorflow Functional API
    input_layer = layers.Input(shape=(1,), dtype=tf.string)
    vec_layer = text_vec(input_layer)
    embedding_layer_model = embedding_layer(vec_layer)
    x = layers.GlobalAveragePooling1D()(embedding_layer_model)
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    model_1 = keras.Model(input_layer, output_layer)

    model_1.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(
        label_smoothing=0.5), metrics=['accuracy'])

    return model_1
