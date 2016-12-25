import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from helper_functions import train_test_stratified

import numpy as np
from subprocess import call
from time import sleep
import pickle

from keras.layers import Dense, GRU, Input, merge
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def batch_generator_multi(X_list, y, batch_size):
    n_splits = len(X_list[0]) // (batch_size)
    X = np.array([np.array(np.array_split(x, n_splits)) for x in X_list])
    y = np.array_split(y, n_splits)

    while True:
        for Xy in zip(X[0], X[1], X[2], y):
            yield ([np.array([x.toarray() for x in Xy[0]]).astype(np.int8),
                    np.array([x.toarray() for x in Xy[1]]).astype(np.int8),
                    np.array([x.toarray() for x in Xy[2]]).astype(np.int8)],
                   Xy[3])


def build_model_shared(input_shape, n_hidden, consume_less="cpu"):
    """
    Build a shared GRU model: (GRU-GRU-GRU)-Dense
    Output from shared layer is summed.
    Taken from https://keras.io/getting-started/functional-api-guide/#shared-layers

    Parameters
    ----------
    input_shape
    n_hidden
    consume_less

    Returns
    -------
    Model
    """

    gru_1 = Input(shape=input_shape)
    gru_2 = Input(shape=input_shape)
    gru_3 = Input(shape=input_shape)

    shared_gru = GRU(n_hidden, consume_less=consume_less)

    encoded_1 = shared_gru(gru_1)
    encoded_2 = shared_gru(gru_2)
    encoded_3 = shared_gru(gru_3)

    merged_vector = merge([encoded_1, encoded_2, encoded_3], mode="sum", concat_axis=-1)
    gru = GRU(n_hidden, consume_less=consume_less)(merged_vector)
    predictions = Dense(1, activation="sigmoid")(gru)

    model = Model(input=[gru_1, gru_2, gru_3], output=predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    # data
    #root_dir = "/storage/brno2/home/jirinovo/_school/pdd/activity/"
    root_dir = ""
    input_file_X = "{}data/data-117k-X-split_part{{}}.npy".format(root_dir)
    input_file_y = "{}data/data-117k-y.npy".format(root_dir)
    model_name = "activity_model-3_shared-117k-grid-150_smiles"

    X_1 = np.load(input_file_X.format("0"))#[0:1000]
    X_2 = np.load(input_file_X.format("1"))#[0:1000]
    X_3 = np.load(input_file_X.format("2"))#[0:1000]
    y = np.load(input_file_y)#[0:1000]

    # network params
    batch_size = 128
    n_epochs = 30
    n_jobs = 8
    consume_less = "cpu"
    n_hidden_list = [16, 32, 64, 128]
    input_shape = X_1[0].shape

    print("X_1, X_2, X_3 shapes:", X_1.shape, X_2.shape, X_2.shape)
    print("y shape:", y.shape)
    print("X_1, X_2, X_3 one sample shapes:", X_1[0].shape, X_2[0].shape, X_3[0].shape)
    print("---------------------------------------------------------------------------------------------------------------")

    train_i, test_i = train_test_stratified(X_1, y, test_size=0.3)

    X_1_train = X_1[train_i]
    X_1_test = X_1[test_i]

    X_2_train = X_2[train_i]
    X_2_test = X_2[test_i]

    X_3_train = X_3[train_i]
    X_3_test = X_3[test_i]

    y_train = y[train_i]
    y_test = y[test_i]

    for n_hidden in n_hidden_list:
        print("\nn_hidden:", n_hidden)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, verbose=5, mode="auto")
        model_checkpoint = ModelCheckpoint("{}results/{}-{}_hidden-best_weights_{{epoch:02d}}_{{val_loss:.5f}}.hdf5".format(root_dir, model_name, n_hidden),
                                           monitor="val_loss", verbose=5, save_best_only=True, mode="auto")
        model = build_model_shared(n_hidden, input_shape)
        history = model.fit_generator(batch_generator_multi([X_1_train, X_2_train, X_3_train], y_train, batch_size),
                                      len(X_1_train),
                                      n_epochs,
                                      validation_data=batch_generator_multi([X_1_test, X_2_test, X_3_test], y_test, batch_size),
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