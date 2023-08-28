# Project Source: https://www.geeksforgeeks.org/sms-spam-detection-using-tensorflow-in-python/
# Dataset Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

import numpy as np
import pandas as pd
import seaborn as sns

import helper
import model_1
import model_2
import model_3

# Reading the data
df = pd.read_csv("./content/spam.csv", encoding='latin-1')
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

# Model 1
model_1 = model_1.build(text_vec, embedding_layer)
# Summary of the model 1
model_1.summary()
history_1 = model_1.fit(X_train, y_train, 5, validation_data=(X_test, y_test), validation_steps=int(0.2 * len(X_test)))
pd.DataFrame(history_1.history).plot()

# plt.show()

# Model 2
model_2 = model_2.build(text_vec, embedding_layer)
# Summary of model 2
model_2.summary()
history_2 = helper.fit_model(model_2, 5, X_train, y_train, X_test, y_test)  # fit the model
pd.DataFrame(history_2.history).plot()

# Model 3
model_3 = model_3.build()
# Summary of model 2
model_3.summary()
history_3 = helper.fit_model(model_3, 5, X_train, y_train, X_test, y_test)
pd.DataFrame(history_3.history).plot()

# Analyzing our Model Performance


baseline_model_results = helper.evaluate_model(baseline_model, X_test_vec, y_test)
model_1_results = helper.evaluate_model(model_1, X_test, y_test)
model_2_results = helper.evaluate_model(model_2, X_test, y_test)
model_3_results = helper.evaluate_model(model_3, X_test, y_test)

total_results = pd.DataFrame({'MultinomialNB Model': baseline_model_results,
                              'Custom-Vec-Embedding Model': model_1_results,
                              'Bidirectional-LSTM Model': model_2_results,
                              'USE-Transfer learning Model': model_3_results}).transpose()

print(f"Model Comparison Total Results:")
print(total_results)

pd.DataFrame(total_results).plot()

# plt.show()
