import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

import torch
from torch.utils.data import Dataset
from rdkit import Chem
import pandas as pd
from sklearn import preprocessing
import pdb
import json
import re

# fmt: off

SMILES_CHARSET = [
    'Br', 'Cl', 'Si', 'Mg', 'Na', 'Al', 'Ca', 'Fe', 'Zn', 'K', 'Ti', 'H', 'He', 'Li', 
    'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'P', 'S', 'Ar', 'V', 'Cr', 'Mn', 'Co', 'Ni', 
    'Cu', 'Ga', 'Ge', 'As', 'Se', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 
    'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 
    'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 
    'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 
    'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', '2', 'o', 'I', 'n', '=', '+', '#', '-', 
    'c', 'B', 'l', '7', 'r', 'S', 's', '4', '6', '[', '5', ']', '3', '(', ')', 
    '1', ' ', '.', '/', '@', '\\'
]
# fmt: on


def tokenize_smiles(smiles):
    pattern = "(%s)" % "|".join(re.escape(token) for token in SMILES_CHARSET)
    tokens = re.findall(pattern, smiles)
    return tokens


class PairedDataset(Dataset):
    def __init__(
        self, dataset_filename, tokenizer, word_max_len=16, smiles_max_len=256
    ):
        self.dataset_df = pd.read_csv(dataset_filename)
        self.dataset_df["Morgan_fingerprint"] = self.dataset_df[
            "Morgan_fingerprint"
        ].apply(json.loads)
        self.tokenizer = tokenizer
        self.word_max_len = word_max_len
        self.smiles_max_len = smiles_max_len
        self.smiles_enc = preprocessing.LabelEncoder().fit(SMILES_CHARSET)

    def __len__(self):
        return len(self.dataset_df)

    def encode_smiles(self, smiles):
        # Tokenize SMILES string into a sequence of tokens
        tokens = tokenize_smiles(smiles)
        encoded_smiles = self.smiles_enc.transform(tokens)
        return encoded_smiles

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        word = row["Word"]
        smiles = row["SMILES"]

        # Encode the word using the BERT tokenizer
        encoded_word = self.tokenizer(
            word,
            padding="max_length",
            truncation=True,
            max_length=self.word_max_len,
            return_tensors="pt",
        )
        input_ids = encoded_word["input_ids"]
        attention_mask = encoded_word["attention_mask"]
        tgt_mask = self._generate_square_subsequent_mask(
            self.word_max_len
        )

        # Encode the SMILES string
        encoded_smiles = self.encode_smiles(smiles)
        encoded_smiles = torch.tensor(encoded_smiles, dtype=torch.float)

        # Compute Morgan fingerprints if needed
        morgan_fingerprints = row["Morgan_fingerprint"]
        morgan_fingerprints = torch.tensor(morgan_fingerprints, dtype=torch.float)

        # Pad SMILES if it's shorter than max_len & cut it off if it's longer
        if len(encoded_smiles) < self.smiles_max_len:
            padding = torch.zeros(self.smiles_max_len - len(encoded_smiles), dtype=torch.long)
            encoded_smiles = torch.cat((encoded_smiles, padding), dim=0)
        elif len(encoded_smiles) > self.smiles_max_len:
            print('yikes! cutting down smiles ', len(encoded_smiles))
            encoded_smiles = encoded_smiles[:self.smiles_max_len]

        return {
            "word": word,
            "word_input_ids": input_ids,
            "word_tgt_mask": tgt_mask,
            "word_attention_mask": attention_mask,
            "smiles": smiles,
            "encoded_smiles": encoded_smiles,
            # "ncoded_smiles_tgt_mask": encoded_smiles_tgt_mask,
            "morgan_fingerprints": morgan_fingerprints,
        }

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


# Example usage
if __name__ == "__main__":
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader

    DATA_FILE = "datasets/playground_dataset.csv"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = PairedDataset(DATA_FILE, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for idx, batch in enumerate(dataloader):
        print("new batch", idx)
        print(batch["word_input_ids"].shape)
        print(batch["word_attention_mask"].shape)
        print(batch["encoded_smiles"].shape)
        print(batch["morgan_fingerprints"].shape)
