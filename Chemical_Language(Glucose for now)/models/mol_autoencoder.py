import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import torch
import torch.nn as nn
import pytorch_lightning as pl
# from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit import Chem
import torch.nn.functional as F

class MoleculeEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(MoleculeEncoder, self).__init__()
        # Purposefully a very simple encoder! 
        # Working from morgan fingerprints which already do quite a beautiful job
        # Tho we importantly don't have the UMAP activity now... 
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class MoleculeDecoder(pl.LightningModule):
#     def __init__(self, latent_dim, output_dim):
#         super(MoleculeDecoder, self).__init__()
#         # TODO: Yikes is a GRU really the best thing here?
#         # We probably want this to be a junction VAE or something like that...
#         self.rnn = nn.GRU(latent_dim, output_dim, batch_first=True)

#     def forward(self, latent_space):
#         latent_space = latent_space.unsqueeze(1).repeat(1, 100, 1)
#         output, _ = self.rnn(latent_space)
#         return output

import pdb


class MoleculeGraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, smiles_len, nchar):
        super(MoleculeGraphDecoder, self).__init__()
        self.smiles_len = smiles_len
        self.nchar = nchar
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, nchar)

    def forward(self, z):
        z = F.relu(self.fc1(z)).unsqueeze(1).repeat(1, self.smiles_len, 1)
        output, _ = self.gru(z)
        output = self.fc2(output)
        pdb.set_trace()
        return output

    def decode_smiles(self, z, charset):
        output = self(z)
        output = torch.argmax(output, dim=-1)
        smiles = "".join([charset[i] for i in output[0].cpu().numpy()])
        return smiles

def compute_smiles_loss(recon_x, x, mu, logvar):
    BCE = F.cross_entropy(recon_x.view(-1, recon_x.size(-1)), x.view(-1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# def main():
#     # Define parameters
#     latent_dim = 128
#     hidden_dim = 256
#     smiles_len = 100
#     nchar = len(moses_charset)

#     # Initialize models
#     vae = MolVAE(rnn_enc_hid_dim=256, enc_nconv=3, encoder_hid=512, z_dim=latent_dim, rnn_dec_hid_dim=256, dec_nconv=3, smiles_len=smiles_len, nchar=nchar)
#     decoder = MolecularGraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, smiles_len=smiles_len, nchar=nchar)

#     # Example latent vector
#     latent_vector = torch.randn(1, latent_dim)

#     # Generate a SMILES string
#     pred_smiles = decoder.decode_smiles(latent_vector, moses_charset)
#     print(f'Predicted SMILES: {pred_smiles}')

#     # Define a simple training loop
#     optimizer = torch.optim.Adam(list(vae.parameters()) + list(decoder.parameters()), lr=1e-3)

#     for epoch in range(10):  # Example training loop
#         optimizer.zero_grad()
#         x = torch.randint(0, nchar, (1, smiles_len))  # Example input
#         recon_x, mu, logvar = vae(x)
#         loss = compute_loss(recon_x, x, mu, logvar)
#         loss.backward()
#         optimizer.step()
#         print(f'Epoch {epoch}, Loss: {loss.item()}')

# if __name__ == "__main__":
#     main()
