import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from helper_functions import train_test_stratified, create_y

import numpy as np
from subprocess import call
from time import sleep
import pickle

from keras.layers import Dense, GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint


def batch_generator(X, y, batch_size):
    n_splits = len(X) // (batch_size)
    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)

    while True:
        for Xy in zip(X, y):
            yield (np.array([x.toarray() for x in Xy[0]]).astype(np.int8),
                   Xy[1])


def build_model(input_shape, n_hidden, consume_less="cpu"):
    model = Sequential([
        GRU(n_hidden, input_shape=input_shape, consume_less=consume_less, return_sequences=True),
        GRU(n_hidden, consume_less=consume_less),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    # data
    root_dir = ""
    #root_dir = "/storage/brno2/home/jirinovo/_school/pdd/activity/"
    X_active_file = "{}data/dna_pol_iota-active-117k-encoded.npy".format(root_dir)
    X_inactive_file = "{}data/zinc-inactive-117k-encoded.npy".format(root_dir)
    model_name = "activity-model-117k-70_30_train_test-150_smiles"

    X_active = np.load(X_active_file)
    X_inactive = np.load(X_inactive_file)

    X = np.concatenate([X_active, X_inactive])
    y = create_y(len(X_active), len(X_inactive))

    del X_active, X_inactive

    train_i, test_i = train_test_stratified(X, y, test_size=0.3)
    X_train = np.copy(X[train_i])
    X_test = np.copy(X[test_i])
    y_train = np.copy(y[train_i])
    y_test = np.copy(y[test_i])

    del X, y

    # train parameters
    batch_size = 128
    n_epochs = 100
    n_jobs = 8
    consume_less = "cpu"
    n_hidden_list = [16, 32, 64, 128]
    input_shape = X_train[0].toarray().shape

    print("train samples:", len(X_train))
    print("test samples:", len(X_test))
    print("one sample shape:", input_shape)

    for n_hidden in n_hidden_list:
        print("\nn_hidden:", n_hidden)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=5, mode="auto")
        model_checkpoint = ModelCheckpoint(
            "{}results/{}-{}_hidden-best_weights_{{epoch:02d}}_{{val_loss:.5f}}.hdf5".format(root_dir, model_name, n_hidden),
            monitor="val_loss", verbose=5, save_best_only=True, mode="auto")
        model = build_model(n_hidden, input_shape)
        history = model.fit_generator(batch_generator(X_train, y_train, batch_size),
                                      len(X_train),
                                      n_epochs,
                                      validation_data=batch_generator(X_test, y_test, batch_size),
                                      nb_val_samples=len(y_train),
                                      nb_worker=n_jobs,
                                      pickle_safe=True,
                                      callbacks=[early_stopping, model_checkpoint])

        model.save("{}models/{}-{}_hidden.hd5".format(root_dir, model_name, n_hidden))

        with open("{}results/{}-history-{}_hidden.pickle".format(root_dir, model_name, n_hidden), mode="bw") as f:
            pickle.dump(history.history, f)

        with open("{}results/{}-history_epoch-{}_hidden.pickle".format(root_dir, model_name, n_hidden), mode="bw") as f:
            pickle.dump(history.epoch, f)

    #sleep(300)
    #call(["poweroff"])