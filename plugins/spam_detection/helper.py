# Let’s create helper functions for compiling, fitting, and evaluating the model performance.
import time

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow import keras


def pre_training():
    # Reading the data
    df = pd.read_csv("/home/will/Projects/AIOpsDemo/plugins/spam_detection/spam.csv", encoding='latin-1')
    df.head()

    df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
    df = df.rename(columns={'v1': 'label', 'v2': 'Text'})
    df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})
    df.head()

    sns.countplot(x=df['label'])
    # plt.show()

    # Find average number of tokens in all sentences
    avg_words_len = round(sum([len(i.split()) for i in df['Text']]) / len(df['Text']))
    print(avg_words_len)

    # Finding Total no of unique words in corpus
    s = set()
    for sent in df['Text']:
        for word in sent.split():
            s.add(word)
    total_words_length = len(s)
    print(total_words_length)

    # Splitting data for Training and testing
    from sklearn.model_selection import train_test_split

    X, y = np.asanyarray(df['Text']), np.asanyarray(df['label_enc'])
    new_df = pd.DataFrame({'Text': X, 'label': y})
    X_train, X_test, y_train, y_test = train_test_split(
        new_df['Text'], new_df['label'], test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import classification_report, accuracy_score

    tfidf_vec = TfidfVectorizer().fit(X_train)
    X_train_vec, X_test_vec = tfidf_vec.transform(X_train), tfidf_vec.transform(X_test)

    baseline_model = MultinomialNB()
    baseline_model.fit(X_train_vec, y_train)

    # Performance of baseline model
    nb_accuracy = accuracy_score(y_test, baseline_model.predict(X_test_vec))
    print(nb_accuracy)
    print(classification_report(y_test, baseline_model.predict(X_test_vec)))

    # Confusion matrix for the baseline model
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Get predictions from the model
    y_pred = baseline_model.predict(X_test_vec)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create a ConfusionMatrixDisplay object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=baseline_model.classes_)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    cm_display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # plt.show()

    # Model Set-Ups

    from tensorflow.keras import layers

    MAXTOKENS = total_words_length
    OUTPUTLEN = avg_words_len

    text_vec = layers.TextVectorization(
        max_tokens=MAXTOKENS,
        standardize='lower_and_strip_punctuation',
        output_mode='int',
        output_sequence_length=OUTPUTLEN
    )
    text_vec.adapt(X_train)

    # Now let’s create an embedding layer

    embedding_layer = layers.Embedding(
        input_dim=MAXTOKENS,
        output_dim=128,
        embeddings_initializer='uniform',
        input_length=OUTPUTLEN
    )

    return baseline_model, text_vec, embedding_layer, X_train, y_train, X_test, y_test, X_test_vec


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


def evaluate_model(model, x, y):
    '''
    evaluate the model and returns accuracy,
    precision, recall and f1-score
    '''
    y_preds = np.round(model.predict(x))
    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)

    model_results_dict = {'accuracy': accuracy,
                          'precision': precision,
                          'recall': recall,
                          'f1-score': f1}

    return model_results_dict


def smoke_test():
    print("Hello World")


def download_dataset():
    print("Placeholder for actually downloading the dataset")
    time.sleep(3)
