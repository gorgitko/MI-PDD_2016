import pandas as pd
from rdkit.Chem import MolToSmiles, MolFromSmiles
from subprocess import call
from time import sleep
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from scipy import sparse
import pickle


def get_noncanonical_smiles_chembl(file_path, output_file_path):
    with open(output_file_path, mode="a") as f:
        f.write("noncanonical_smiles\tcanonical_smiles")

    data = pd.read_csv(file_path, delimiter=";")

    with open(output_file_path, mode="a") as f:
        for smiles in data["smiles"]:
            smiles = smiles.strip().replace(" ", "")
            mol = MolFromSmiles(smiles)
            if mol:
                new_smiles = MolToSmiles(mol)
                if smiles != new_smiles:
                    f.write("\n{}\t{}".format(smiles, new_smiles))


def get_noncanonical_smiles_zinc(file_path, output_file_path):
    with open(output_file_path, mode="a") as f:
        f.write("noncanonical_smiles\tcanonical_smiles")

    n_nc_smiles_total = 0
    for i in range(17):
        if i < 10:
            file_path_part = file_path.format(0, i)
        else:
            file_path_part = file_path.format("", i)

        print("Processing file:", file_path_part)
        with open(file_path_part, mode="r") as f:
            data_part = pd.Series(f.readlines())

        with open(output_file_path, mode="w") as f:
            n_nc_smiles_part = 0
            for smiles in data_part:
                smiles = smiles.strip().replace(" ", "")
                mol = MolFromSmiles(smiles)
                if mol:
                    new_smiles = MolToSmiles(mol)
                    if smiles != new_smiles:
                        f.write("\n{}\t{}".format(smiles, new_smiles))
                        n_nc_smiles_part += 1
        n_nc_smiles_total += n_nc_smiles_part

        print("Done! Added {} new non-canonical SMILES ({} total).".format(n_nc_smiles_part, n_nc_smiles_total))


def encode(arr, coding_arr):
    """
    Encode list of SMILES's charcodes to 0-1 matrix where each row is one ASCII code and each col is one of the possible
    SMILES's charcodes. If charcode is 0, then whole row will be 0s, since charcode 0 means a padded character.
    Saved as a NumPy array of SciPi's Compressed Sparse Row matrices (scipy.sparse.csr_matrix).

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
    2D matrix
    """
    coding_arr = coding_arr.flatten()
    matrices = np.array([])
    unknown_charcodes = []

    for row in arr:
        row = row.flatten()
        table = np.zeros(shape=(row.size, coding_arr.size), dtype=np.int8)
        for i, x in enumerate(np.nditer(row)):
            if x:
                try:
                    table[i, np.where(coding_arr == x)[0][0]] = 1
                except IndexError:
                    unknown_charcodes.append(x)
        table = sparse.csr_matrix(table, dtype=np.int8)
        matrices = np.append(matrices, table)

    print("Unknown charcodes:", unknown_charcodes)
    return np.array(matrices)

def save_train_test_split(input_file_path, output_file, n_samples=0, test_size=0.3, longest_smiles=0, random_state=1,
                          smiles_charcodes_file="data/smiles_charcodes.npy", delimiter="\t"):
    smiles_codes = np.load(smiles_charcodes_file)
    data = pd.read_csv(input_file_path, delimiter=delimiter)
    data["noncanonical_smiles"] = data["noncanonical_smiles"].apply(str.strip)
    data["canonical_smiles"] = data["canonical_smiles"].apply(str.strip)
    #data_active["smiles"] = data_active["smiles"].astype("str")

    if n_samples:
        data = data.sample(n=n_samples, random_state=random_state)

    if longest_smiles:
        data = data[(data["noncanonical_smiles"].apply(len) <= longest_smiles) & (data["canonical_smiles"].apply(len) <= longest_smiles)]
    else:
        longest_smiles = pd.Series(data["noncanonical_smiles"], data["canonical_smiles"]).apply(len).max()

    data["noncanonical_smiles"] = data["noncanonical_smiles"].apply(lambda x: [ord(c) for c in x])
    data["canonical_smiles"] = data["canonical_smiles"].apply(lambda x: [ord(c) for c in x])

    X_train, X_test, y_train, y_test = train_test_split(data["noncanonical_smiles"], data["canonical_smiles"], test_size=test_size, random_state=random_state)

    X_train = sequence.pad_sequences(X_train.values, maxlen=longest_smiles)
    X_train = encode(X_train, smiles_codes)
    np.save("{}-X_train".format(output_file), X_train)
    del X_train

    X_test = sequence.pad_sequences(X_test.values, maxlen=longest_smiles)
    X_test = encode(X_test, smiles_codes)
    np.save("{}-X_test".format(output_file), X_test)
    del X_test

    y_train = sequence.pad_sequences(y_train.values, maxlen=longest_smiles)
    print(y_train[0])
    #y_train = encode(y_train, smiles_codes)
    np.save("{}-y_train".format(output_file), y_train)
    del y_train

    y_test = sequence.pad_sequences(y_test.values, maxlen=longest_smiles)
    #y_test = encode(y_test, smiles_codes)
    np.save("{}-y_test".format(output_file), y_test)
    del y_test


#get_noncanonical_smiles_chembl("../molecular-weight/data/mw_chembl_all.csv", "data/noncanonical-smiles_chembl.txt")
#get_noncanonical_smiles_zinc("/home/jirka/temp/zinc/smiles/zinc.smiles.part{}{}", "data/noncanonical-smiles_zinc.txt")
#save_train_test_split("data/noncanonical-smiles_chembl.txt", "data/smiles_100k_150-length.npy", longest_smiles=150, n_samples=100000)
#save_train_test_split("data/noncanonical-smiles_chembl.txt", "data/smiles_100k_150-length_sparse", longest_smiles=150, n_samples=100000)
save_train_test_split("data/noncanonical-smiles_chembl.txt", "data/smiles_100k_150-length_y-charcodes", longest_smiles=150, n_samples=100000)

#smiles_codes = np.load("data/smiles_charcodes.npy")
#smiles_encoded = encode(np.array([[ord(c) for c in "CCNCCOCNON(CCON)CCNOCONC"]]), smiles_codes)
#print(smiles_encoded[0][0, :])
#print("bytes:", smiles_encoded.nbytes)
#smiles_encoded_sparse = sparse.csr_matrix(smiles_encoded[0])
#print(np.all(smiles_encoded[0] == smiles_encoded_sparse))
#print(np.all(smiles_encoded[0] == smiles_encoded_sparse.toarray()))
#print(smiles_encoded_sparse.toarray()[0, :])
#print(smiles_encoded_sparse.data.nbytes)
#print(smiles_encoded_sparse[0:2, :])
#sleep(180)
#call(["poweroff"])