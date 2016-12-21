import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
import matplotlib.pyplot as plt


def sqlite_to_csv(file_path, output_file_path):
    connection = sqlite3.connect(file_path)
    cursor = connection.cursor()
    cursor.execute(r"""SELECT
      cs.canonical_smiles, cp.full_mwt
    FROM
      compound_properties AS cp
    JOIN
      compound_structures AS cs ON cs.molregno = cp.molregno
    """)
    tables = cursor.fetchall()

    with open(output_file_path, mode="w") as f:
        f.write("smiles;mw\n")
        for tbl in tables:
            f.write("{};{}\n".format(tbl[0], tbl[1]))

def encode(arr, coding_arr):
    """
    Encode list of SMILES's charcodes to 0-1 matrix where each row is one ASCII code and each col is one of the possible
    SMILES's charcodes. If charcode is 0, then whole row will be 0s, since charcode 0 means a padded character.

    x     35 36 37 ... 67 ... 78 79 ... 121
    ---------------------------------------
    0    | 0  0  0 ...  0 ...  0  0 ...   0
    0    | 0  0  0 ...  0 ...  0  0 ...   0
    0    | 0  0  0 ...  0 ...  0  0 ...   0
    .    | .  .  . ...  . ...  .  . ...   .
    .    | .  .  . ...  . ...  .  . ...   .
    .    | .  .  . ...  . ...  .  . ...   .
    C->67| 0  0  0 ...  1 ...  0  0 ...   0
    C->67| 0  0  0 ...  1 ...  0  0 ...   0
    N->78| 0  0  0 ...  0 ...  1  0 ...   0
    O->79| 0  0  0 ...  0 ...  0  1 ...   0

    Parameters
    ----------
    arr
    coding_arr

    Returns
    -------
    3D tensor
    """
    coding_arr = coding_arr.flatten()
    matrices = []

    for row in arr:
        row = row.flatten()
        table = np.zeros(shape=(row.size, coding_arr.size))
        for i, x in enumerate(np.nditer(row)):
            if x:
                table[i, np.where(coding_arr == x)[0][0]] = 1
        matrices.append(table)

    return np.array(matrices)

def save_train_test_split(file_path, full_size, output_file, test_size=0.3, longest_smiles=0, random_state=1,
                          smiles_codes_file="smiles_charcodes.npy"):
    smiles_codes = np.load(smiles_codes_file)
    data = pd.read_csv(file_path, delimiter=";")

    #data["smiles"].apply(len).plot.hist(bins=30)
    #plt.show()

    if longest_smiles:
        data = data[data["smiles"].apply(len) <= longest_smiles]
    else:
        longest_smiles = data["smiles"].apply(len).max()

    data = data.sample(n=full_size, random_state=random_state)

    data["smiles"] = data["smiles"].apply(lambda x: [ord(c) for c in x])

    X_train, X_test, y_train, y_test = train_test_split(data["smiles"], data["mw"], test_size=test_size, random_state=random_state)

    #mm = np.mean(np.concatenate([y_train, y_test]))
    #ss = np.std(np.concatenate([y_train, y_test]))
    #print(y_train)
    #print(mm, ss)
    mm = np.mean(y_train)
    ss = np.std(y_train)
    y_train = (y_train - mm) / ss

    mm = np.mean(y_test)
    ss = np.std(y_test)
    y_test = (y_test - mm) / ss

    X_train = sequence.pad_sequences(X_train.values, maxlen=longest_smiles)
    X_train = encode(X_train, smiles_codes)
    X_test = sequence.pad_sequences(X_test.values, maxlen=longest_smiles)
    X_test = encode(X_test, smiles_codes)
    #y_train = y_train.values.reshape(len(y_train), 1)
    #y_test = y_test.values.reshape(len(y_test), 1)

    np.save("{}-X_train".format(output_file), X_train)
    np.save("{}-X_test".format(output_file), X_test)
    np.save("{}-y_train".format(output_file), y_train)
    np.save("{}-y_test".format(output_file), y_test)

#sqlite_to_csv("/home/jirka/temp/chembl_22.db", "data/mw_chembl_all.csv")
#save_train_test_split("data/mw_chembl_all.csv", 10000, "data/data-10k-3k_test")
#save_train_test_split("data/mw_chembl_all.csv", 100000, "data/data-100k-30k_test")
save_train_test_split("data/mw_chembl_all.csv", 30000, "data/data-30k-9k_test-150_smiles", longest_smiles=150)
#save_train_test_split("data/mw_chembl_all.csv", 10000, "data/data-10k-3k_test-150_smiles", longest_smiles=150)