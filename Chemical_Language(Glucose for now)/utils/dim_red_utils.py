import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))

from sklearn.decomposition import PCA
import umap
import numpy as np


# TODO: put intermediary dim back to 300
def pca_to_umap(df, col_name, intermediary_dim=50, final_dim=30):
    fingerprints = np.array(df[col_name].tolist())
    pca = PCA(n_components=intermediary_dim)  # Reduce to 100 dimensions
    pcaed = pca.fit_transform(fingerprints)

    reducer = umap.UMAP(
        n_neighbors=5, min_dist=0.1, n_components=final_dim, random_state=42
    )
    embedding = reducer.fit_transform(pcaed)
    return embedding
