import numpy as np
from subprocess import call
from time import sleep

from keras.layers import Dense, GRU, Activation, Embedding, Reshape, Input, LSTM
from keras.models import Sequential


def batch_generator(X, y, batch_size):
    n_splits = len(X) // (batch_size - 1)
    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)
    for i in range(len(X)):
        X_batch = []
        y_batch = []
        for ii in range(len(X[i])):
            X_batch.append(X[i][ii].toarray())
            y_batch.append(y[i][ii].toarray())
        yield (np.array(X_batch), np.array(y_batch))


def batch_generator_y_charcodes(X, y, batch_size):
    n_splits = len(X) // (batch_size - 1)
    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)
    for i in range(len(X)):
        X_batch = []
        y_batch = []
        for ii in range(len(X[i])):
            X_batch.append(X[i][ii].toarray())
            y_batch.append(y[i][ii])
        yield (np.array(X_batch), np.array(y_batch))


def custom_objective(y_true, y_pred):
    print(y_true)
    print(y_pred)


batch_size = 128
epochs = 100
#input_file = "data/smiles_100k_150-length_sparse.npy-{}.npy"
input_file = "data/smiles_100k_150-length_y-charcodes-{}.npy"
model_file = "models/canonical-smiles_100k.hd5"

X_train = np.load(input_file.format("X_train"))
X_test = np.load(input_file.format("X_test"))
y_train = np.load(input_file.format("y_train"))
y_test = np.load(input_file.format("y_test"))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""
i = 0
for x in batch_generator(X_train, y_train, batch_size):
    if i == 5:
        print(x[0].shape)
        print(x[1].shape)
        print(x[0])
        print("--------------\n\n\n")
        print(x[1])
        break
    i += 1

exit()
"""

#gen = batch_generator_y_charcodes(X_train, y_train, batch_size=batch_size)
#sample = next(gen)
#print(sample[1])

model = Sequential([
    GRU(150, input_shape=X_train[0].shape, consume_less="cpu", return_sequences=False),
    Activation("linear"),
    #Reshape(X_train[0].shape)
    #Dense(1, activation="sigmoid")
    ])

model.compile(#loss="binary_crossentropy",
            loss=custom_objective,
              optimizer="adam")

model.fit_generator(batch_generator_y_charcodes(X_train, y_train, batch_size),
                    len(X_train),
                    epochs,
                    validation_data=batch_generator_y_charcodes(X_test, y_test, batch_size),
                    nb_val_samples=len(y_train),
                    nb_worker=4,
                    pickle_safe=True)

model.save(model_file)
