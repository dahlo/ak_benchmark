

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the image classifier.
clf = ak.ImageClassifier(
            max_trials=1,
            seed=1234567890)

# Feed the image classifier with training data.
clf.fit(x_train, y_train, epochs=3)

# Predict with the best model.
#predicted_y = clf.predict(x_test)
#print(predicted_y)


# Evaluate the best model with testing data.
#print(clf.evaluate(x_test, y_test))
