import numpy as np
import scipy
import matplotlib.pyplot as plt
from numba import jit

from sklearn.cross_decomposition import CCA

@jit
def time_warp(costs):
    dtw = np.zeros_like(costs)
    dtw[0,1:] = np.inf
    dtw[1:,0] = np.inf
    eps = 1e-4
    for i in range(1,costs.shape[0]):
        for j in range(1,costs.shape[1]):
            dtw[i,j] = costs[i,j] + min(dtw[i-1,j],dtw[i,j-1],dtw[i-1,j-1])
    return dtw

def align_from_distances(distance_matrix, debug=False):
    # for each position in spectrum 1, returns best match position in spectrum2
    # using monotonic alignment
    dtw = time_warp(distance_matrix)

    i = distance_matrix.shape[0]-1
    j = distance_matrix.shape[1]-1
    results = [0] * distance_matrix.shape[0]
    while i > 0 and j > 0:
        results[i] = j
        i, j = min([(i-1,j),(i,j-1),(i-1,j-1)], key=lambda x: dtw[x[0],x[1]])

    if debug:
        visual = np.zeros_like(dtw)
        visual[range(len(results)),results] = 1
        plt.matshow(visual)
        plt.show()

    return results

def get_all_alignments(dataset, feature_functions, weights=None, return_aligned_features=False):
    # note returned aligned features only currently returns features from the last function

    assert len(feature_functions) >= 1, 'expected non-empty list'
    if weights is not None:
        assert len(weights) == len(feature_functions)
    alignments = []
    aligned_features = []
    for example in dataset:
        f1, f2 = feature_functions[0](example)
        distances = scipy.spatial.distance.cdist(f1, f2, metric='euclidean')
        if weights is not None:
            distances *= weights[0]
        for i in range(1,len(feature_functions)):
            f1, f2 = feature_functions[i](example)
            dist = scipy.spatial.distance.cdist(f1, f2, metric='euclidean')
            if weights is not None:
                dist *= weights[i]
            distances += dist

        alignment = align_from_distances(distances)

        alignments.append(alignment)
        aligned_features.append((f1, f2[alignment]))

    if return_aligned_features:
        return aligned_features
    else:
        return alignments

def get_cca_transform(dataset, feature_function):
    aligned_features = get_all_alignments(dataset, [feature_function], return_aligned_features=True)

    cca = CCA(n_components=15)
    X = np.concatenate([s for s,v in aligned_features],0)
    Y = np.concatenate([v for s,v in aligned_features],0)
    cca.fit(X,Y)

    def cca_transform(example):
        f1, f2 = feature_function(example)
        return cca.transform(f1, f2)

    return cca_transform

