from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import helper

# Model -2 Bidirectional LSTM
print("RUNNING Model -2 Bidirectional LSTM")


def build(text_vec, embedding_layer):
    input_layer = layers.Input(shape=(1,), dtype=tf.string)
    vec_layer = text_vec(input_layer)
    embedding_layer_model = embedding_layer(vec_layer)
    bi_lstm = layers.Bidirectional(layers.LSTM(
        64, activation='tanh', return_sequences=True))(embedding_layer_model)
    lstm = layers.Bidirectional(layers.LSTM(64))(bi_lstm)
    flatten = layers.Flatten()(lstm)
    dropout = layers.Dropout(.1)(flatten)
    x = layers.Dense(32, activation='relu')(dropout)
    output_layer = layers.Dense(1, activation='sigmoid')(x)
    model_2 = keras.Model(input_layer, output_layer)

    helper.compile_model(model_2)  # compile the model

    return model_2