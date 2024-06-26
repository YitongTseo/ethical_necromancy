import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics.pairwise import euclidean_distances
import pdb
import einops
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW


class WordEncoder(pl.LightningModule):
    def __init__(self, latent_dim, bert_dim=768):
        super(WordEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # TODO: The bert model is not frozen! We can fine tune it!
        # But perhaps we don't want that? Perhaps we want to freeze it... we shall see...
        # self.fc = nn.Linear(bert_dim, latent_dim)
        # No activation function!

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze()
            )
        outputs = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        return outputs  # self.fc(outputs)


import torch.nn.functional as F
from torch.nn import Transformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)  # Registering as buffer

    def forward(self, x):
        # pdb.set_trace()
        # (Pdb) [int(np.round(i)) for i in self.pe[1][0]]
        return x + self.pe[: x.size(0), :]



class WordDecoder(pl.LightningModule):
    def __init__(self, latent_dim, vocab_size, max_len=16):
        super(WordDecoder, self).__init__()
        self.max_len = max_len
        # self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.positional_encoding = PositionalEncoding(latent_dim, max_len)
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2.resize_token_embeddings(
            vocab_size
        )  # Resize token embeddings to match your vocab size
        self.fc_out = nn.Linear(latent_dim, vocab_size)

    def forward(self, latent_space, tgt, tgt_mask=None):
        latent_space_unsqueezed = latent_space.unsqueeze(0).repeat(self.max_len, 1, 1)
        latent_space_unsqueezed = self.positional_encoding(latent_space_unsqueezed)

        # Embed the target using GPT-2's embeddings
        tgt_emb = self.gpt2.transformer.wte(tgt).squeeze()
        tgt_emb = self.positional_encoding(tgt_emb.permute(1, 0, 2))

        # Decode the embeddings with GPT-2
        # TODO: we probably want to deal with the attention_mask eventually
        outputs = self.gpt2(
            inputs_embeds=latent_space_unsqueezed
        )  # , attention_mask=tgt_mask)
        logits = outputs.logits.permute(
            1, 0, 2
        )  # Convert back to [batch_size, tgt_len, vocab_size]
        return logits

    def decode(self, latent, tokenizer):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            batch_size = latent.size(0)
            device = latent.device
            # Apply positional encoding to latent space
            latent_space_unsqueezed = latent.unsqueeze(0).repeat(self.max_len, 1, 1)
            latent_space_unsqueezed = self.positional_encoding(latent_space_unsqueezed)

            # Initialize the input with the [CLS] token (assuming 101 is the ID for [CLS])
            input_ids = torch.tensor(
                [tokenizer.cls_token_id] * batch_size, device=device
            ).unsqueeze(1)
            finished_sequences = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )
            for i in range(1, self.max_len):
                # Get the latent vector for the current time step
                current_latent = latent_space_unsqueezed[i : i + 1]

                # Decode using GPT-2 with the positional encoded latent space
                outputs = self.gpt2(
                    inputs_embeds=current_latent,
                    attention_mask=(~finished_sequences).unsqueeze(1),
                )
                next_token = outputs.logits.argmax(dim=-1).squeeze().reshape((-1, 1))
                input_ids = torch.cat((input_ids, next_token), dim=1)

                # Update finished_sequences
                # The model is not perfect and sometimes outputs 0's instead of ending correctly
                finished_sequences = (
                    finished_sequences
                    | ((next_token == tokenizer.sep_token_id)).squeeze()
                    | ((next_token == 0)).squeeze()
                )
                # Break if all sequences are finished
                if finished_sequences.all():
                    break

            word_reconstruction_decoded = input_ids.tolist()
            decoded_sentences = [
                tokenizer.decode(token_id, skip_special_tokens=True)
                for token_id in word_reconstruction_decoded
            ]
            return decoded_sentences, word_reconstruction_decoded
