# Project Source: https://www.geeksforgeeks.org/sms-spam-detection-using-tensorflow-in-python/
# Dataset Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

import pandas as pd

import helper
import model_1
import model_2
import model_3

baseline_model, text_vec, embedding_layer, x_train, y_train, x_test, y_test, x_test_vec = helper.pre_training()

# Model 1
model_1 = model_1.build(text_vec, embedding_layer)
# Summary of the model 1
model_1.summary()
history_1 = model_1.fit(x_train, y_train, 5, validation_data=(x_test, y_test), validation_steps=int(0.2 * len(x_test)))
pd.DataFrame(history_1.history).plot()

# plt.show()

# Model 2
model_2 = model_2.build(text_vec, embedding_layer)
# Summary of model 2
model_2.summary()
history_2 = helper.fit_model(model_2, 5, x_train, y_train, x_test, y_test)  # fit the model
pd.DataFrame(history_2.history).plot()

# Model 3
model_3 = model_3.build()
# Summary of model 2
model_3.summary()
history_3 = helper.fit_model(model_3, 5, x_train, y_train, x_test, y_test)
pd.DataFrame(history_3.history).plot()

# Analyzing our Model Performance


baseline_model_results = helper.evaluate_model(baseline_model, x_test_vec, y_test)
model_1_results = helper.evaluate_model(model_1, x_test, y_test)
model_2_results = helper.evaluate_model(model_2, x_test, y_test)
model_3_results = helper.evaluate_model(model_3, x_test, y_test)

total_results = pd.DataFrame({'MultinomialNB Model': baseline_model_results,
                              'Custom-Vec-Embedding Model': model_1_results,
                              'Bidirectional-LSTM Model': model_2_results,
                              'USE-Transfer learning Model': model_3_results}).transpose()

print(f"Model Comparison Total Results:")
print(total_results)

winner = total_results[total_results['accuracy'] == total_results['accuracy'].max()]

print(f"The winner is {winner}")

pd.DataFrame(total_results).plot()

# plt.show()
