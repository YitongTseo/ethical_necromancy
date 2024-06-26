import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import torch
import torch.nn as nn
import pytorch_lightning as pl
import pdb
import torch.nn.functional as F


class JointAutoencoder(pl.LightningModule):
    def __init__(self, word_encoder, word_decoder, mol_encoder, mol_decoder):
        super(JointAutoencoder, self).__init__()
        self.word_encoder = word_encoder
        self.word_decoder = word_decoder
        self.mol_encoder = mol_encoder
        self.mol_decoder = mol_decoder
        self.criterion = nn.MSELoss()
        self.word_reconstruction_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        word_input_ids,
        word_attention_mask,
        fingerprints,
        word_tgt_mask,
        # encoded_smiles,
        # encoded_smiles_tgt_mask,
    ):
        word_latent = self.word_encoder(word_input_ids, word_attention_mask)
        molecule_latent = self.mol_encoder(fingerprints)

        # Not going to lie, I don't really understand this TGT = word_input_ids part...
        # Seems we feed it the ground truth during training? Isn't that cheating?
        word_reconstruction = self.word_decoder(
            word_latent, word_input_ids, word_tgt_mask
        )
        # TODO: figure out this mol_decoder
        molecule_reconstruction = None  # self.mol_decoder(molecule_latent)

        return (
            word_latent,
            molecule_latent,
            word_reconstruction,
            molecule_reconstruction,
        )

    def training_step(self, batch, batch_idx):
        # batch['word']
        # batch['smiles']
        word_input_ids = batch["word_input_ids"]
        word_attention_mask = batch["word_attention_mask"]
        encoded_smiles = batch["encoded_smiles"]
        word_tgt_mask = batch["word_tgt_mask"]
        # encoded_smiles_tgt_mask = batch["encoded_smiles_tgt_mask"]
        fingerprints = batch["morgan_fingerprints"]
        word_latent, molecule_latent, word_reconstruction, molecule_reconstruction = (
            self(
                word_input_ids,
                word_attention_mask,
                fingerprints,
                word_tgt_mask,
            )
        )
        # Convert word_input_ids to one-hot encoded tensor
        vocab_size = word_reconstruction.size(-1)
        word_input_ids_one_hot = F.one_hot(
            word_input_ids.squeeze(), num_classes=vocab_size
        ).float()

        word_input_ids_one_hot = word_input_ids_one_hot.contiguous().view(
            -1, vocab_size
        )  # [batch_size * max_len, vocab_size]
        word_reconstruction = word_reconstruction.contiguous().view(
            -1, vocab_size
        )  # [batch_size * max_len, vocab_size]
        # word_reconstruction.argmax(dim=-1).shape
        # word_input_ids_one_hot.argmax(dim=-1).shape

        word_loss = self.word_reconstruction_criterion(
            word_reconstruction, word_input_ids_one_hot
        )

        # molecule_loss = self.criterion(molecule_reconstruction, molecule_inputs)
        alignment_loss = self.criterion(word_latent, molecule_latent)

        total_loss = word_loss + alignment_loss  # + molecule_loss
        self.log("combined_loss", total_loss)
        self.log("word_loss", word_loss)
        # self.log("molecule_loss", molecule_loss)
        self.log("alignment_loss", alignment_loss)
        return total_loss

    # def testing_step(self, batch, batch_idx, tokenizer):
    #     word_input_ids = batch["word_input_ids"]
    #     word_attention_mask = batch["word_attention_mask"]
    #     encoded_smiles = batch["encoded_smiles"]
    #     word_tgt_mask = batch["word_tgt_mask"]
    #     # words = batch['word']
    #     # smiles = batch['smiles']

    #     # encoded_smiles_tgt_mask = batch["encoded_smiles_tgt_mask"]
    #     fingerprints = batch["morgan_fingerprints"]
    #     word_latent, molecule_latent, word_reconstruction, molecule_reconstruction = (
    #         self(
    #             word_input_ids,
    #             word_attention_mask,
    #             fingerprints,
    #             word_tgt_mask,
    #         )
    #     )
    #     # Convert word_input_ids to one-hot encoded tensor
    #     vocab_size = word_reconstruction.size(-1)
    #     word_input_ids_one_hot = F.one_hot(
    #         word_input_ids.squeeze(), num_classes=vocab_size
    #     ).float()

    #     word_input_ids_one_hot = word_input_ids_one_hot.contiguous().view(
    #         -1, vocab_size
    #     )  # [batch_size * max_len, vocab_size]
    #     word_reconstruction = word_reconstruction.contiguous().view(
    #         -1, vocab_size
    #     )  # [batch_size * max_len, vocab_size]

    #     pdb.set_trace()
    #     batch_size = len(word_input_ids)
    #     word_reconstruction_decoded = word_reconstruction.argmax(dim=-1).view(
    #         batch_size, -1
    #     )
    #     word_input_ids_from_1Hot = word_input_ids_one_hot.argmax(dim=-1).view(
    #         batch_size, -1
    #     )
    #     assert word_input_ids_from_1Hot == word_input_ids

    #     decoded_sentences = [
    #         tokenizer.decode(token_id, skip_special_tokens=True)
    #         for token_id in word_reconstruction_decoded.tolist()
    #     ]
    #     return decoded_sentences

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
