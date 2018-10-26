# compare softmax_outputs w/ target indicator matrix
import numpy as np

prediction_labels = np.argmax(softmax_outputs, axis=1)

target_labels = np.argmax(target_indicator, axis = 1)

prediction_labels = [1, 0, 2, 1]

target_labels = [1, 2, 2, 0]

accuracy = sum(prediction_labels == target_labels)/len(prediction_labels)