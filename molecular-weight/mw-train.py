import pandas as pd
import numpy as np
from subprocess import call
from time import sleep

from keras.layers import Dense, GRU, Activation, Embedding, Reshape, Input, LSTM
from keras.models import Sequential
#from keras.utils.visualize_util import plot


batch_size = 128
epochs = 100
#input_file = "data/data-10k-3k_test-{}.npy"
#model_file = "models/mw-model-10k.hd5"
#input_file = "data/data-10k-3k_test-150_smiles-{}.npy"
#model_file = "models/mw-model-10k-150_smiles.hd5"
input_file = "data/data-30k-9k_test-150_smiles-{}.npy"
model_file = "models/mw-model-30k-150_smiles.hd5"

X_train = np.load(input_file.format("X_train"))
X_test = np.load(input_file.format("X_test"))
y_train = np.load(input_file.format("y_train"))
y_test = np.load(input_file.format("y_test"))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model = Sequential([
    GRU(128, input_shape=X_train[0].shape, consume_less="cpu"),
    Dense(1)
    ])

#plot(model, to_file="model.png", show_layer_names=True, show_shapes=True)

model.compile(loss="mean_squared_error",
              optimizer="adam")

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, y_test))
model.save(model_file)

sleep(300)
call(["poweroff"])