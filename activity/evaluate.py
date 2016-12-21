import numpy as np
from sklearn.metrics.classification import classification_report, accuracy_score
from keras.models import load_model


input_file = "data/data-23k-70_30_train_test-150_smiles-{}.npy"
model = load_model("models/activity-model-23k-70_30_train_test-150_smiles.hd5")
X_test = np.load(input_file.format("X_test"))
y_test = np.load(input_file.format("y_test"))

y_predicted_test = np.round(model.predict(X_test, verbose=0).flatten())

print(model.summary())
print("keras accuraccy:", model.evaluate(X_test, y_test, verbose=0))
print("scikit accuracy:", accuracy_score(y_test, y_predicted_test))
print(classification_report(y_test, y_predicted_test))
