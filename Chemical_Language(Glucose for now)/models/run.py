import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from models.dataset import PairedDataset, SMILES_CHARSET
from models.mol_autoencoder import MoleculeEncoder, MoleculeGraphDecoder
from models.word_autoencoder import WordEncoder, WordDecoder
from models.joint_autoencoder import JointAutoencoder
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

DATA_FILE = "datasets/playground_dataset.csv"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

paired_loader = DataLoader(
    PairedDataset(DATA_FILE, tokenizer, smiles_max_len=256), batch_size=32, shuffle=True
)

word_encoder = WordEncoder(latent_dim=768)
word_decoder = WordDecoder(latent_dim=768, vocab_size=tokenizer.vocab_size)
mol_encoder = MoleculeEncoder(input_dim=2048, hidden_dim=1024, latent_dim=768)
mol_decoder = MoleculeGraphDecoder(
    latent_dim=768, hidden_dim=1024, smiles_len=256, nchar=len(SMILES_CHARSET)
)

model = JointAutoencoder(
    word_encoder,
    word_decoder,
    mol_encoder,
    mol_decoder,
)
# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='combined_loss',  # Monitor training loss to save the best model
    dirpath='checkpoints/',  # Directory to save the checkpoints
    filename='joint-autoencoder-{epoch:02d}-{combined_loss:.2f}',  # Checkpoint filename format
    save_top_k=2,
    mode='min',  # Save the model with the minimum training loss
)
logger = TensorBoardLogger("tb_logs", name="joint_autoencoder")

num_devices = 1 if not torch.cuda.is_available() else torch.cuda.device_count()
trainer = pl.Trainer(
    max_epochs=100,
    devices=num_devices,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    callbacks=[checkpoint_callback],
    logger=logger,
    log_every_n_steps=1
)
trainer.fit(model, paired_loader)
