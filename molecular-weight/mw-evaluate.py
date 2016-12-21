import numpy as np
from sklearn.metrics import r2_score
from keras.models import load_model


input_file = "data/data-30k-9k_test-150_smiles-{}.npy"
model = load_model("models/mw-model-10k-150_smiles.hd5")
X_test = np.load(input_file.format("X_test"))
y_test = np.load(input_file.format("y_test"))

mm = np.mean(y_test)
ss = np.std(y_test)
y_test = (y_test - mm) / ss

print(model.summary())
print("MSE:", model.evaluate(X_test, y_test))
y_predicted_test = model.predict(X_test)
#print("test:", y_test[0:5])
#print("predicted:", y_predicted_test[0:5])
print("R2:", r2_score(y_test, y_predicted_test))