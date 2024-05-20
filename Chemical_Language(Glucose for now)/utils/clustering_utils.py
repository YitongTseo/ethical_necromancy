import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute().parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute().parent))


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.optimize import linear_sum_assignment
import json
import pandas as pd
from tqdm import tqdm
import pdb
from utils.dim_red_utils import pca_to_umap


def kmeans_clustering(datapoints, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(datapoints)
    return clusters


def divisive_clustering(datapoints, percentile=10, min_cluster_size=None):
    distance_threshold = np.percentile(euclidean_distances(datapoints), percentile)
    return divisive_clustering_helper(datapoints, distance_threshold, min_cluster_size)


def divisive_clustering_helper(
    datapoints, distance_threshold, min_cluster_size, debug=False
):
    def check_cluster_distances(datapoints, clusters, threshold):
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            cluster_points = datapoints[clusters == cluster]
            if cluster_points.shape[0] < 2:
                continue
            distances = euclidean_distances(cluster_points)
            np.fill_diagonal(
                distances, 0
            )  # Ensure diagonal is zero to ignore self-distances
            if np.any(distances > threshold):
                return False, cluster
        return True, None

    clusters = np.zeros(len(datapoints))
    if debug:
        print(
            " initial cluster sizes: ",
            [np.sum(clusters == i) for i in np.unique(clusters)],
        )
    while True:
        all_meet_criteria, cluster_to_split = check_cluster_distances(
            datapoints, clusters, distance_threshold
        )
        if all_meet_criteria:
            break

        cluster_points = datapoints[clusters == cluster_to_split]
        kmeans = KMeans(n_clusters=2, random_state=42)
        sub_clusters = kmeans.fit_predict(cluster_points)

        if min_cluster_size is not None:
            # Check if any sub-cluster is smaller than min_cluster_size
            sub_cluster_sizes = [
                np.sum(sub_clusters == i) for i in np.unique(sub_clusters)
            ]
            if debug:
                print(
                    "sub_cluster_sizes ",
                    sub_cluster_sizes,
                    "\nmin_cluster_size ",
                    min_cluster_size,
                )
            if any(size < min_cluster_size for size in sub_cluster_sizes):
                if debug:
                    print("so d o we break?")
                break

        new_cluster_labels = np.where(
            sub_clusters == 0, cluster_to_split, np.max(clusters) + 1
        )
        clusters[clusters == cluster_to_split] = new_cluster_labels

    if debug:
        print(
            "returned cluster sizes is: ",
            [np.sum(clusters == i) for i in np.unique(clusters)],
        )
    return clusters


def compute_centroids(data, labels, cluster_labels):
    centroids = []
    for label in cluster_labels:
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def hungarian_mapping(dataset_1, dataset_2):
    similarity_matrix = euclidean_distances(dataset_1, dataset_2)
    cost_matrix = similarity_matrix  # Use distances as cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return mapping


def get_ds2_clusters(dataset_2, percentile, min_cluster_num, min_cluster_size):
    # Make sure none of the ds2_clusters have fewer datapoints than a ds1_cluster
    # Otherwise we won't have an all datapoint mapping from dataset_1
    ds2_clusters = divisive_clustering(
        dataset_2, percentile, min_cluster_size
    )

    # Ensure ds2_clusters has at least as many clusters as ds1_clusters
    num_ds2_clusters = len(np.unique(ds2_clusters))
    while num_ds2_clusters < min_cluster_num:
        ds2_cluster_sizes = [
            np.sum(ds2_clusters == cluster) for cluster in np.unique(ds2_clusters)
        ]
        largest_cluster = np.argmax(ds2_cluster_sizes)
        largest_cluster_points = dataset_2[ds2_clusters == largest_cluster]

        sub_clusters = divisive_clustering(
            largest_cluster_points,
            percentile,
            min_cluster_size,
        )
        sub_cluster_sizes = [np.sum(sub_clusters == i) for i in np.unique(sub_clusters)]

        if (
            any(size < min_cluster_size for size in sub_cluster_sizes)
            or len(sub_cluster_sizes) == 1
        ):
            # This check... this unfortunate tradeoff between wanting at least as many clusters as ds1
            # and wanting the clusters to all be above the max ds1 cluster size
            # is why we can't have nice things...
            # is why it's hard / (as far as I see right now) impossible to give certainty to the
            # all points in dataset_1 getting a mapping partner.
            break

        new_cluster_labels = np.where(
            sub_clusters == 0, largest_cluster, np.max(ds2_clusters) + 1
        )
        ds2_clusters[ds2_clusters == largest_cluster] = new_cluster_labels
        num_ds2_clusters = len(np.unique(ds2_clusters))

    return ds2_clusters

def size_mapping(_datset_1, _dataset_2, ds1_clusters, ds2_clusters):
    """
    Just maps the largeset cluster in dataset_1 to the largest cluster in datset_2, etc.
    """
    # print('we in the size mapping?')
    # ds1_clusters[ds1_clusters == 0]
    ds1_cluster_sizes = {
        cluster: np.sum(ds1_clusters == cluster) for cluster in np.unique(ds1_clusters)
    }
    ds2_cluster_sizes = {
        cluster: np.sum(ds2_clusters == cluster) for cluster in np.unique(ds2_clusters)
    }

    sorted_ds1_cluster_sizes = sorted(
        ds1_cluster_sizes.items(), key=lambda item: item[1], reverse=True
    )
    sorted_ds2_cluster_sizes = sorted(
        ds2_cluster_sizes.items(), key=lambda item: item[1], reverse=True
    )

    mapping = {
        int(sorted_ds1_cluster_sizes[idx][0]): int(sorted_ds2_cluster_sizes[idx][0])
        for idx in range(
            min(len(sorted_ds1_cluster_sizes), len(sorted_ds2_cluster_sizes))
        )
    }
    return mapping


def recursive_divisive_clustering(
    dataset_1,
    dataset_2,
    depth=0,
    max_depth=5,
    ds1_percentile=10,
    bottom_out_size=10,
    debug=False,
):
    assert len(dataset_1) <= len(dataset_2), "len(dataset_1) must be <= len(dataset_2)"
    if (
        len(dataset_1) <= bottom_out_size
        or len(dataset_2) <= bottom_out_size
        or depth >= max_depth
    ):
        return None, None, hungarian_mapping(dataset_1, dataset_2)

    ds1_clusters = divisive_clustering(dataset_1, ds1_percentile)
    max_ds1_cluster_size = max(
        np.sum(ds1_clusters == cluster) for cluster in np.unique(ds1_clusters)
    )
    num_ds1_clusters = len(np.unique(ds1_clusters))

    # Make sure none of the ds2_clusters have fewer datapoints than a ds1_cluster
    # & make sure there are at least as many ds2_clusters as ds1_clusters (hard!)

    # NOTE: A heuristic that helps is setting percentile lower for ds2 relative to ds1
    ds2_clusters = get_ds2_clusters(
        dataset_2,
        percentile=ds1_percentile / 2,
        min_cluster_num=num_ds1_clusters,
        min_cluster_size=max_ds1_cluster_size,
    )
    print(
        "num ds1_clusters ",
        num_ds1_clusters,
        " num ds2_clusters ",
        len(np.unique(ds2_clusters)),
    )

    if debug and depth == 0:
        print(len(ds1_clusters))
        print(ds1_clusters)
        print(len(ds2_clusters))
        print(ds2_clusters)

    ds1_centroids = compute_centroids(dataset_1, ds1_clusters, np.unique(ds1_clusters))
    ds2_centroids = compute_centroids(dataset_2, ds2_clusters, np.unique(ds2_clusters))

    hungarian_mapping(ds1_centroids, ds2_centroids)

    final_mapping = {}
    if depth == 0:
        iterator = tqdm(cluster_mapping.items(), desc="Top-Level Clustering")
    else:
        iterator = cluster_mapping.items()

    for ds1_cluster, ds2_cluster in iterator:
        ds1_cluster_points = dataset_1[ds1_clusters == ds1_cluster]
        ds2_cluster_points = dataset_2[ds2_clusters == ds2_cluster]

        _, _, sub_mapping = recursive_divisive_clustering(
            ds1_cluster_points, ds2_cluster_points, depth + 1, max_depth, ds1_percentile
        )

        for k, v in sub_mapping.items():
            ds1_idx = np.where((dataset_1 == ds1_cluster_points[k]).all(axis=1))[0][0]
            ds2_idx = np.where((dataset_2 == ds2_cluster_points[v]).all(axis=1))[0][0]
            final_mapping[ds1_idx] = ds2_idx

        if debug and depth == 0:
            print(
                "len(ds1_cluster_points)",
                len(ds1_cluster_points),
                " len(ds2_cluster_points)",
                len(ds2_cluster_points),
                " ",
                sub_mapping,
                final_mapping,
            )
    # For the sake of pretty graphs also return the top level clusters
    return ds1_clusters, ds2_clusters, final_mapping


def recursive_kmeans_clustering(
    dataset_1,
    dataset_2,
    depth=0,
    max_depth=5,
    initial_num_clusters=2,
    bottom_out_size=10,
):
    # assert len(dataset_1) <= len(dataset_2), "len(dataset_1) must be <= len(dataset_2)"
    if (
        len(dataset_1) <= bottom_out_size
        or len(dataset_2) <= bottom_out_size
        or depth >= max_depth
    ):
        return None, None, hungarian_mapping(dataset_1, dataset_2)

    ds1_clusters = kmeans_clustering(
        dataset_1, n_clusters=int(initial_num_clusters )#/ 2**depth)
    )
    ds2_clusters = kmeans_clustering(
        dataset_2, n_clusters=int(initial_num_clusters) #/ 2**depth)
    )

    ds1_centroids = compute_centroids(dataset_1, ds1_clusters, np.unique(ds1_clusters))
    ds2_centroids = compute_centroids(dataset_2, ds2_clusters, np.unique(ds2_clusters))

    # cluster_mapping = size_mapping(dataset_1, dataset_2, ds1_clusters, ds2_clusters) 
    cluster_mapping = hungarian_mapping(ds1_centroids, ds2_centroids)

    final_mapping = {}
    if depth == 0:
        iterator = tqdm(cluster_mapping.items(), desc="Top-Level Clustering")
    else:
        iterator = cluster_mapping.items()

    for ds1_cluster, ds2_cluster in iterator:
        ds1_cluster_points = dataset_1[ds1_clusters == ds1_cluster]
        ds2_cluster_points = dataset_2[ds2_clusters == ds2_cluster]

        _, _, sub_mapping = recursive_kmeans_clustering(
            ds1_cluster_points,
            ds2_cluster_points,
            depth + 1,
            max_depth,
            initial_num_clusters,
        )

        for k, v in sub_mapping.items():
            ds1_idx = np.where((dataset_1 == ds1_cluster_points[k]).all(axis=1))[0][0]
            ds2_idx = np.where((dataset_2 == ds2_cluster_points[v]).all(axis=1))[0][0]
            final_mapping[ds1_idx] = ds2_idx

    # For the sake of pretty graphs also return the top level clusters
    return ds1_clusters, ds2_clusters, final_mapping


# words_df = pd.read_csv("playground_BERT_embeddings.csv")
# mols_df = pd.read_csv("playground_enamine_fingerprints.csv")

# mols_df = mols_df[:200]
# words_df = words_df[:50]

# words_df["BERT_Embedding"] = words_df["BERT_Embedding"].apply(json.loads)
# mols_df["Morgan_fingerprint"] = mols_df["Morgan_fingerprint"].apply(json.loads)

# molecular_fingerprints = pca_to_umap(mols_df, col_name="Morgan_fingerprint")
# bert_embeddings = pca_to_umap(words_df, col_name="BERT_Embedding")

# mapping = recursive_clustering(bert_embeddings, molecular_fingerprints)

# # pdb.set_trace()
# # pass
