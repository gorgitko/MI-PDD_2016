import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from helper_functions import encode_smiles, save_smiles_charcodes, canonize_smiles

import pandas as pd
import numpy as np


def save_active_compounds(input_file, output_file, encode=True, longest_smiles=0, smiles_charcodes_file="data/smiles_charcodes.npy",
                          smiles_col="CANONICAL_SMILES", delimiter="\t"):
    """
    Save X_active compounds from ChEMBL CSV file.

    Parameters
    ----------
    input_file
    output_file
    encode
        If True, encode SMILES to one-hot matrices.
    longest_smiles
        How long should be longest SMILES. If len(smiles) < longest_smiles, it gets padded with 0's (resp. its ASCII charcodes).
    smiles_charcodes_file
        .npy file containing list of all possible ASCII charcodes of SMILES.

    Returns
    -------
    numpy.array
        If encode, contains one-hot matrices (scipy.sparse.csr_matrix) of SMILES. Otherwise array of SMILES strings.
    """

    compounds = pd.read_csv(input_file, delimiter=delimiter)
    #compounds = compounds[compounds["STANDARD_UNITS"].isin(["nM", "uM"])]
    #compounds = compounds[compounds["STANDARD_TYPE"].isin(["Kd", "Potency"])]
    compounds = compounds[smiles_col]
    compounds = compounds.astype("str")
    compounds = compounds.apply(canonize_smiles)
    compounds = compounds[compounds != "invalid"]

    if encode:
        compounds = encode_smiles(compounds, np.load(smiles_charcodes_file), longest_smiles=longest_smiles)

    np.save(output_file, compounds)
    return compounds


def save_inactive_compounds(input_file, output_file, n_compounds, n_files=17, encode=True, longest_smiles=0,
                            smiles_charcodes_file="data/smiles_charcodes.npy"):
    """
    Save X_inactive compounds from multiple files containing SMILES from ZINC.

    Parameters
    ----------
    input_file
    output_file
    n_compounds
    n_files
    encode
        If True, encode SMILES to one-hot matrices.
    longest_smiles
        How long should be longest SMILES. If len(smiles) < longest_smiles, it gets padded with 0's (resp. its ASCII charcodes).
    smiles_charcodes_file
        .npy file containing list of all possible ASCII charcodes of SMILES.

    Returns
    -------
    numpy.array
        If encode, contains one-hot matrices (scipy.sparse.csr_matrix) of SMILES. Otherwise array of SMILES strings.
    """
    n_per_file = n_compounds // n_files
    compounds = pd.Series()
    for i in range(n_files):
        if i < 10:
            file_path_part = input_file.format(0, i)
        else:
            file_path_part = input_file.format("", i)
        print("Processing input_file {}/{}:".format(i + 1, n_files), file_path_part)
        with open(file_path_part, mode="r") as f:
            data_part = [x.strip() for x in f.readlines()]
        data_part = pd.Series(data_part)
        compounds = compounds.append(data_part.sample(n=n_per_file))

    compounds = compounds.apply(canonize_smiles)
    compounds = compounds[compounds != "invalid"]

    if encode:
        compounds = encode_smiles(compounds, np.load(smiles_charcodes_file), longest_smiles=longest_smiles)

    np.save(output_file, compounds)
    return compounds


if __name__ == "__main__":
    #save_active_compounds("data/dna_pol_iota-X_active-117k.csv", "data/dna_pol_iota-X_active-117k-encoded", longest_smiles=150, encode=True)
    #save_inactive_compounds("/home/jirka/temp/zinc/smiles/zinc.smiles.part{}{}", "data/zinc-X_inactive-117k-encoded", 116723, longest_smiles=150, encode=True)
    #save_active_compounds("data/dna_pol_iota-active-117k.csv", "data/dna_pol_iota-active-117k-smiles", longest_smiles=150, encode=False)
    #save_inactive_compounds("/home/jirka/temp/zinc/smiles/zinc.smiles.part{}{}", "data/zinc-inactive-117k-smiles", 116723, longest_smiles=150, encode=False)
    pass
