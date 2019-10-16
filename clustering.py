import logging
import pickle
from collections import defaultdict

import numpy as np
import networkx as nx

from deprecated import deprecated
from tqdm import tqdm
from string import ascii_uppercase
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import plotly.graph_objs as go
import plotly.io as pio


# Preamble
logging.basicConfig(level=logging.INFO)
SEED = 42


def best_kmeans(X, max_range=np.arange(2, 11), criterion='silhouette'):
    """
    Returns the best K-Means clustering given the data, a range of K values, and a K-selection criterion.

    :param X: usage matrix (made of usage vectors)
    :param max_range: range within the number of clusters should lie
    :param criterion: K-selection criterion: 'silhouette' or 'calinski'
    :return: best_model: KMeans model (sklearn.cluster.Kmeans) with best clustering according to the criterion
             scores: list of tuples (k, s) indicating the clustering score s obtained using k clusters
    """
    assert criterion in ['silhouette', 'calinski', 'harabasz', 'calinski-harabasz']

    best_model, best_score = None, -1
    scores = []

    for k in max_range:
        if k < X.shape[0]:
            kmeans = KMeans(n_clusters=k, random_state=SEED)
            cluster_labels = kmeans.fit_predict(X)

            if criterion == 'silhouette':
                score = silhouette_score(X, cluster_labels)
            else:
                score = calinski_harabasz_score(X, cluster_labels)

            scores.append((k, score))

            # if two clusterings yield the same score, keep the one that results from a smaller K
            if score > best_score:
                best_model, best_score = kmeans, score

    return best_model, scores


def to_one_hot(y, num_classes=None):
    """
    Transform a list of categorical labels into the list of corresponding one-hot vectors.
    E.g. [2, 3, 1] -> [[0,0,1,0], [0,0,0,1], [0,1,0,0]]

    :param y: N-dimensional array of categorical class labels
    :param num_classes: the number C of distinct labels. Inferred from `y` if unspecified.
    :return: N-by-C one-hot label matrix
    """
    if num_classes:
        K = num_classes
    else:
        K = np.max(y) + 1

    one_hot = np.zeros((len(y), K))

    for i in range(len(y)):
        one_hot[i, y[i]] = 1

    return one_hot


def usage_distribution(predictions, time_labels):
    """

    :param predictions: The clustering predictions
    :param time_labels:
    :return:
    """
    if predictions.ndim > 2:
        raise ValueError('Array of cluster probabilities has too many dimensions: {}'.format(predictions.ndim))
    if predictions.ndim == 1:
        predictions = to_one_hot(predictions)

    label_set = sorted(list(set(time_labels)))
    t2i = {t: i for i, t in enumerate(label_set)}

    n_clusters = predictions.shape[1]
    n_periods = len(label_set)
    usage_distr = np.zeros((n_clusters, n_periods))

    for pred, t in zip(predictions, time_labels):
        usage_distr[:, t2i[t]] += pred.T

    return usage_distr


def make_usage_matrices(dict_path, mode='concat', usages_out=None, ndims=768):
    """
    Take as input a usage dictionary containing single usage vectors (and their metadata)
    and return a dictionary that maps lemmas to usage matrices (and the same metadata).

    :param dict_path: path to the pickled usage dictionary
    :param mode: 'sum' or 'concatenation'
    :param usages_out: dictionary to map words to usage matrices (and their metadata)
    :param ndims: dimensionality of usage vectors
    :return: dictionary mapping words to usage matrices and their metadata: w -> (Uw, contexts, positions, time_labels)
    """
    assert mode in ['sum', 'concat', 'concatenation', 'cat']

    with open(dict_path, 'rb') as f:
        usages_in = pickle.load(f)

    if usages_out is None:
        usages_out = {}
        for w in usages_in:
            usages_out[w] = (np.empty((0, ndims)), [], [], [])

    for w in tqdm(usages_in):
        for (vec, context, pos_in_context, decade) in usages_in[w]:
            usages_out[w] = (
                np.row_stack((usages_out[w][0], vec)),
                usages_out[w][1] + [context],
                usages_out[w][2] + [pos_in_context],
                usages_out[w][3] + [decade]
            )

    if mode == 'sum':
        for w in usages_out:
            Uw, contexts, positions, t_labels = usages_out[w]
            Uw_layerwise = Uw.reshape((Uw.shape[0], Uw.shape[1], ndims, -1))
            Uw_sum = np.sum(Uw_layerwise, axis=3)
            usages_out[w] = Uw_sum, contexts, positions, t_labels

    return usages_out


def cluster_usages(Uw, method='kmeans', k_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return the best clustering model for a usage matrix.

    :param Uw: usage matrix
    :param method: K-Means or Gaussian Mixture Model ('kmeans' or 'gmm')
    :param k_range: range of possible K values (number of clusters)
    :param criterion: K selection criterion; depends on clustering method
    :return: best clustering model
    """
    # standardize usage matrix by removing the mean and scaling to unit variance
    X = preprocessing.StandardScaler().fit_transform(Uw)

    # get best model according to a K-selection criterion
    if method == 'kmeans':
        best_model, scores = best_kmeans(X, k_range, criterion=criterion)
    elif method == 'gmm':
        raise NotImplementedError('Gaussian Mixture Model not yet implemented!')
    else:
        raise ValueError('Invalid method "{}". Choose "kmeans" or "gmm".'.format(method))

    return best_model


def obtain_clusterings(usages, out_path, method='kmeans', k_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return and save dictionary mapping lemmas to their best clustering model, given a method-criterion pair.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param out_path: output path to store clustering models
    :param method: K-Means or Gaussian Mixture Model ('kmeans' or 'gmm')
    :param k_range: range of possible K values (number of clusters)
    :param criterion: K selection criterion; depends on clustering method
    :return: dictionary mapping lemmas to their best clustering model
    """
    clusterings = {}  # dictionary mapping lemmas to their best clustering
    for w in usages:
        Uw, _, _, _ = usages[w]
        clusterings[w] = cluster_usages(Uw, method, k_range, criterion)

    with open(out_path, 'wb') as f:
        pickle.dump(clusterings, file=f)

    return clusterings


def plot_usage_distribution(usages, clusterings, out_dir, normalized=False):
    """
    Save plots of probability- or frequency-based usage distributions.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param clusterings: dictionary mapping lemmas to their best clustering model
    :param out_dir: output directory for plots
    :param normalized: whether to normalize usage distributions
    """
    for word in clusterings:
        _, _, _, t_labels = usages[word]
        best_model = clusterings[word]

        # create usage distribution based on clustering results
        usage_distr = usage_distribution(best_model.labels_, t_labels)
        if normalized:
            usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)

        # create a bar plot with plotly
        data = []
        for i in range(usage_distr.shape[0]):
            data.insert(0, go.Bar(
                y=usage_distr[i, :],
                name='usage {}'.format(ascii_uppercase[i])
            ))
        layout = go.Layout(title=word,
                           xaxis=dict(
                               ticktext=list(np.arange(1910, 2009, 10)),
                               tickvals=list(np.arange(10))),
                           barmode='stack')

        fig = go.Figure(data=data, layout=layout)
        pio.write_image(fig, '{}/{}_{}.pdf'.format(
            out_dir,
            word,
            'prob' if normalized else 'freq'))


def clustering_intersection(models):
    """
    Given a list of clusterings L = (C', C'', ...) of the same unsupervised data set X = (x_1, x_2, ..., x_n),
    return the clustering intersection I such that I(x_i) == I(x_j) iff C(x_i) == C(x_j) for all C in L.

    :param models: list L of clustering objects (with attribute `labels_`)
    :return: list of labels according to the clustering intersection I.
             Instances that do not belong to any intersection are given label -1.
    """
    labels = []
    # ensure we are working with clusterings of the same data set
    for model in models:
        if labels:
            assert len(model.labels_) == len(labels[-1]), "Different number of instances across models!"
        labels.append(model.labels_)

    n_nodes = len(labels[0])
    G = nx.Graph()  # create graph G=(N,V) to store common clustering decisions
    G.add_nodes_from(list(np.arange(n_nodes)))  # N is the set of usages

    # E = {(u_i, u_j) | C(u_i) == C(u_j) for all C in L}
    for i in np.arange(n_nodes):
        for j in np.arange(n_nodes):
            if i == j:
                continue
            if all([label[i] == label[j] for label in labels]):
                G.add_edge(i, j)

    # The clustering intersection I is the list of connected components of G
    meta_labels = np.full(shape=n_nodes, fill_value=-1, dtype=np.int)
    for meta_label, cc in enumerate(nx.connected_components(G)):
        for i in cc:
            meta_labels[i] = meta_label

    return meta_labels


# todo: from wordpieces back to words
def parse_snippet(snippet):
    return snippet


def collect_snippets(usages, clusterings):
    """
    Collect usage snippets and organise them according to their usage type and time interval.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param clusterings: dictionary mapping lemmas to their best clustering model
    :return: dictionary mapping (lemma, cluster, time) triplets to lists of usage snippets
    """
    snippets = {}  # (lemma, cluster_id, time_interval) -> [(<s>, ..., <\s>), (<s>, ..., <\s>), ...]

    for word in usages:
        snippets[word] = defaultdict(lambda: defaultdict(list))

        _, contexts, _, t_labels = usages[word]
        cl_labels = clusterings[word].labels_

        for context, cl, t in zip(contexts, cl_labels, t_labels):
            snippets[word][cl][t].append(parse_snippet(context))

    return snippets


@deprecated('Compute clustering instersection instead!')
def percentage_agreement(models):
    labels = []
    for model in models:
        if labels:
            assert len(model.labels_) == len(labels[-1]), "Different number of instances across models!"
        labels.append(model.labels_)

    print(labels)

    agreement = 0.
    tot = 0.
    for i in np.arange(len(labels[0])):
        for j in np.arange(len(labels[0])):
            if i == j:
                continue
            # print([label[i] == label[j] for label in labels])
            if all([label[i] == label[j] for label in labels]):
                agreement += 1
            if all([label[i] != label[j] for label in labels]):
                agreement += 1
            tot += 1

    return agreement / tot
