# Let’s create helper functions for compiling, fitting, and evaluating the model performance.
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


def compile_model(model):
    '''
    simply compile the model with adam optimzer
    '''
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])


def fit_model(model, epochs, x_train, y_train,
              x_test, y_test):
    '''
    fit the model with given epochs, train
    and test data
    '''
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        validation_steps=int(0.2 * len(x_test)))
    return history


def evaluate_model(model, X, y):
    '''
    evaluate the model and returns accuracy,
    precision, recall and f1-score
    '''
    y_preds = np.round(model.predict(X))
    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)

    model_results_dict = {'accuracy': accuracy,
                          'precision': precision,
                          'recall': recall,
                          'f1-score': f1}

    return model_results_dict
