import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore", message="not removing hydrogen atom without neighbors")
warnings.filterwarnings("ignore", message="not removing hydrogen atom with dummy atom neighbors")

def get_single_molecule_embedding(smiles, radius=5, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:    
        # Remove explicit hydrogens
        mol = Chem.RemoveHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        return list(fp)
    return None

def process_and_append(
    df, start_idx=0, chunk_size=1000, output_file="final_processed_data.csv"
):
    for start in tqdm(range(start_idx, len(df), chunk_size)):
        end = start + chunk_size
        chunk = df.iloc[start:end].copy()  # Using copy to safely modify the data
        chunk["Morgan_fingerprint"] = chunk["SMILES"].apply(
            get_single_molecule_embedding
        )
        mode = "a" if start > 0 else "w"  # 'w' for write on first chunk, 'a' for append on subsequent chunks
        header = start == 0  # Write header only for the first chunk
        chunk.to_csv(output_file, mode=mode, header=header, index=False)

def check_progress(output_file):
    """Check if the output file already exists and determine the last processed index"""
    try:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            df = pd.read_csv(output_file)
            last_index_processed = df.index[-1]
            return last_index_processed + 1
        else:
            return 0
    except FileNotFoundError:
        return 0
