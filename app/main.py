from unicodedata import name
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

class_names = ["healthy", "suspicious", "sick"]

train_data = pd.read_csv("./data/train.csv", delimiter=";", names=["#", "gender", "age", "pbf", "systolic blood pressure", "diastolic blood pressure", "pulse", "glucose", "saturation", "type"])

train_features = train_data.copy()
train_features.pop('#')
train_labels = train_features.pop('type')

train_features = np.array(train_features)

train_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(1)
])

train_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['accuracy'])

train_model.fit(train_features, train_labels, epochs=10)

# -- Train

test_data = pd.read_csv("./data/test.csv", delimiter=";", names=["#", "gender", "age", "pbf", "systolic blood pressure", "diastolic blood pressure", "pulse", "glucose", "saturation", "type"])

test_features = test_data.copy()
test_features.pop('#')
test_labels = test_features.pop('type')

test_features = np.array(test_features)

test_loss, test_acc = train_model.evaluate(test_features, test_labels, verbose=2)

print('\nTest accuraccy:', test_acc)

probability_model = tf.keras.Sequential([train_model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_features)

print(predictions)

