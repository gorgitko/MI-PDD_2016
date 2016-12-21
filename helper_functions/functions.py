import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import warnings
from rdkit.Chem import MolFromSmiles, MolToSmiles
from sklearn.model_selection import StratifiedShuffleSplit


def encode_smiles(X, coding_arr, longest_smiles=0):
    """
    Encode list of SMILES's charcodes to 0-1 matrix where each row is one ASCII code and each col is one of the possible
    SMILES's charcodes. If charcode is 0, then whole row will be 0s, since charcode 0 means a padded character.
    Return a NumPy array of SciPi's Compressed Sparse Row matrices (scipy.sparse.csr_matrix).

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
    X : pd.Series
        Series of SMILES.
    coding_arr : np.array
        NumPy array of SMILES ASCII charcodes.
    longest_smiles : int
        How long should be longest SMILES. If len(smiles) < longest_smiles, it gets padded with 0's (resp. its ASCII charcodes).

    Returns
    -------
    numpy.array
        NumPy array of SciPy's sparse 2D matrices (scipy.sparse.csr_matrix).
    """

    coding_arr = coding_arr.flatten()
    matrices = np.array([])
    unknown_chars = []

    if longest_smiles:
        X = X[X.apply(len) <= longest_smiles]
    else:
        longest_smiles = X.apply(len).max()

    X = X.apply(lambda x: [ord(c) for c in x]).values
    X = pad_sequences(X, maxlen=longest_smiles)

    for row in X:
        row = row.flatten()
        table = np.zeros(shape=(row.size, coding_arr.size), dtype=np.int8)
        for i, x in enumerate(np.nditer(row)):
            if x:
                try:
                    table[i, np.where(coding_arr == x)[0][0]] = 1
                except IndexError:
                    unknown_chars.append((x, chr(x)))
        table = csr_matrix(table, dtype=np.bool)
        matrices = np.append(matrices, table)

    if unknown_chars:
        warnings.warn("These characters aren't in coding_arr: {}".format(unknown_chars))

    return np.array(matrices)


def save_smiles_charcodes(output_file="smiles_charcodes", chars=None):
    """
    Convert all possible SMILES characters to list of their corresponding ASCII codes.

    Parameters
    ----------
    output_file
    chars

    Returns
    -------
    np.array
    """

    if not chars:
        chars = {
            '-',
            '=', '#', '*', '/', '\\', '%',
            '.', '(', ')', '[', ']',
            '{', '}', '@', '+', '0',
            '1', '2', '3', '4', '5',
            '6', '7', '8', '9', 'a',
            'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E',
            'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y',
            'Z'
         }

    char_codes = np.array([list(set(map(ord, chars)))])
    np.save(output_file, char_codes)


def canonize_smiles(smiles):
    mol = MolFromSmiles(smiles)
    if mol:
        return MolToSmiles(mol)
    else:
        return "invalid"


def train_test_stratified(X, y, test_size=0.3):
    """
    Split the data to train-test sets so there are same amounts of classes.

    Parameters
    ----------
    X
    y
    test_size

    Returns
    -------
    (list, list)
        Indexes of train and test data.
    """

    i_list = np.array(list(StratifiedShuffleSplit(test_size=test_size).split(X, y)))
    id = np.random.choice(len(i_list), 1)
    train_i = i_list[id][0][0]
    test_i = i_list[id][0][1]
    return train_i, test_i


def create_y(n_active, n_inactive):
    y_active = np.zeros((n_active))
    y_active.fill(1)
    y_inactive = np.zeros((n_inactive))
    return np.concatenate([y_active, y_inactive]).astype(np.int8)

"""
chars = ['B', 'C', 'N', 'O', 'S', 'P', 'F', 'C', 'l', 'B', 'r', 'I', 'b', 'c', 'n', 'o', 's', 'p', '[', ']',
         '(', ')', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'H', 'H', 'e', 'L', 'i', 'B', 'e', 'B', 'C', 'N',
         'O', 'F', 'N', 'e', 'N', 'a', 'M', 'g', 'A', 'l', 'S', 'i', 'P', 'S', 'C', 'l', 'A', 'r', 'K', 'C', 'a', 'S', 'c', 'T',
         'i', 'V', 'C', 'r', 'M', 'n', 'F', 'e', 'C', 'o', 'N', 'i', 'C', 'u', 'Z', 'n', 'G', 'a', 'G', 'e', 'A', 's',
         'S', 'e', 'B', 'r', 'K', 'r', 'R', 'b', 'S', 'r', 'Y', 'Z', 'r', 'N', 'b', 'M', 'o', 'T', 'c', 'R', 'u', 'R',
         'h', 'P', 'd', 'A', 'g', 'C', 'd', 'I', 'n', 'S', 'n', 'S', 'b', 'T', 'e', 'I', 'X', 'e', 'C', 's', 'B', 'a', 'H',
         'f', 'T', 'a', 'W', 'R', 'e', 'O', 's', 'I', 'r', 'P', 't', 'A', 'u', 'H', 'g', 'T', 'l', 'P', 'b', 'B', 'i',
         'P', 'o', 'A', 't', 'R', 'n', 'F', 'r', 'R', 'a', 'R', 'f', 'D', 'b', 'S', 'g', 'B', 'h', 'H', 's', 'M', 't',
         'D', 's', 'R', 'g', 'C', 'n', 'F', 'l', 'L', 'v', 'L', 'a', 'C', 'e', 'P', 'r', 'N', 'd', 'P', 'm', 'S', 'm',
         'E', 'u', 'G', 'd', 'T', 'b', 'D', 'y', 'H', 'o', 'E', 'r', 'T', 'm', 'Y', 'b', 'L', 'u', 'A', 'c', 'T', 'h',
         'P', 'a', 'U', 'N', 'p', 'P', 'u', 'A', 'm', 'C', 'm', 'B', 'k', 'C', 'f', 'E', 's', 'F', 'm', 'M', 'd', 'N',
         'o', 'L', 'r', 'b', 'c', 'n', 'o', 'p', 's', 's', 'e', 'a', 's', '@', '-', '=', '#', '$', ':', '/', '\\', '.', '+',
         '%']
"""