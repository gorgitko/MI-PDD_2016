import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from helper_functions import encode_smiles, create_y

import pandas as pd
import numpy as np
from pprint import pprint


def find_brackets(smiles):
    """
    Find indexes of the first matching brackets ( "(" and ")" ). It doesn't check if all brackets are valid, i.e. complete.

    Parameters
    ----------
    smiles

    Returns
    -------
    list
        Index of first and second matching bracket.
    """

    indexes = []
    n_brackets = 0
    for i, char in enumerate(smiles):
        if char == "(":
            if n_brackets == 0:
                indexes.append(i)
            n_brackets += 1
        elif char == ")":
            n_brackets -= 1
            if n_brackets == 0:
                indexes.append(i)
                break
    return indexes


def split_smiles(smiles, min_len=5, join_char=""):
    """
    Split SMILES to three parts. The second part will be the content between first matching brackets.
    But this content must contain >= min_len chars, so some brackets are skipped.
    E.g. if min_len=5 and smiles="CC(C)CC(NCOOCC)CCCN(NC)", then return will be ["CC(C)CC", "NCOOCC", "CCCN(NC)"]
    If none bracket-content is >= min_len chars, then return will be [smiles, "", ""]

    Parameters
    ----------
    smiles
    min_len
    join_char
        The character which will be appended to end, resp. beginning of splited SMILES (it should indicate that there
        was bracket. I.e. smiles="CC(C)CC(NCOOCC)CCCN(NC)", join_char="*" -> ["CC(C)CC*", "*NCOOCC*", "*CCCN(NC)"]

    Returns
    -------
    str
    """

    smiles_parts_list = []
    smiles_original = smiles

    while True:
        indexes = find_brackets(smiles)

        if indexes:
            smiles_parts = [smiles[0:indexes[0]],
                            smiles[indexes[0]+1:indexes[1]],
                            smiles[indexes[1]+1:]]
            smiles_parts_list.append(smiles_parts)
            if len(smiles_parts[1]) >= min_len:
                break
            else:
                smiles = smiles_parts[2]
        else:
            break

    if indexes:
        smiles_new = "".join(["{}({})".format(x[0], x[1]) for x in smiles_parts_list[0:-1]])
        last_part = smiles_parts_list[-1]
        return [smiles_new + last_part[0] + join_char, join_char + last_part[1] + join_char, join_char + last_part[2]]
    else:
        return [smiles_original, "", ""]


def get_splits(X, longest_smiles, min_len=5):
    X_splits = []
    for smiles in X:
        smiles_split = split_smiles(smiles, min_len=min_len)
        if all([False for x in map(len, smiles_split) if x > longest_smiles]):
            X_splits.append([smiles_split[0], smiles_split[1], smiles_split[2]])
    return X_splits


if __name__ == "__main__":
    #root_dir = "/storage/brno2/home/jirinovo/_school/pdd/activity/"
    root_dir = ""
    X_active_file = "{}data/dna_pol_iota-active-117k-smiles.npy".format(root_dir)
    X_inactive_file = "{}data/zinc-inactive-117k-smiles.npy".format(root_dir)
    X_file = "{}data/data-1k-X-50_smiles-split_part{{}}".format(root_dir)
    y_file = "{}data/data-1k-y-50_smiles".format(root_dir)
    smiles_charcodes = np.load("{}data/smiles_charcodes.npy".format(root_dir))
    longest_smiles = 50
    min_len = 5  # minimal length of bracket content
    n_samples = 1000

    X_active = np.load(X_active_file)
    X_inactive = np.load(X_inactive_file)

    if n_samples:
        X_active = np.random.choice(X_active, size=n_samples)
        X_inactive = np.random.choice(X_inactive, size=n_samples)

    print("n X_active:", len(X_active))
    print("n X_inactive:", len(X_inactive))

    X_splits_active = get_splits(X_active, longest_smiles, min_len=min_len)
    X_splits_inactive = get_splits(X_inactive, longest_smiles, min_len=min_len)

    for i in range(3):
        print("\nencoding split:", i)
        X_split = np.concatenate([[x[i] for x in X_splits_active], [x[i] for x in X_splits_inactive]])
        X_split = encode_smiles(pd.Series(X_split), smiles_charcodes, longest_smiles=longest_smiles)
        np.save(X_file.format(i), X_split)

    y = create_y(len(X_splits_active), len(X_splits_inactive))
    np.save(y_file, y)
