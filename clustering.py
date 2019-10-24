import itertools
import logging
import pickle
import random

import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

from collections import defaultdict
from deprecated import deprecated
from tqdm import tqdm
from string import ascii_uppercase
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from transformers import BertTokenizer

FIGURE_EIGHT_FIELDS = ['lemma', 'id_a', 'id_b', 'cluster_a', 'cluster_b', 'time_a', 'time_b', 'a', 'b']
FIGURE_EIGHT_FIELDS_TEST = ['fieldname_gold', 'fieldname_gold_reason', '_golden']
SEED = 42

logging.basicConfig(level=logging.INFO)
np.random.seed(SEED)


def best_kmeans(X, max_range=np.arange(2, 11), criterion='silhouette'):
    """
    Return the best K-Means clustering given the data, a range of K values, and a K-selection criterion.

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


class GMM(object):
    """
    A Gaussian Mixture Model object, with its number of components, AIC and BIC scores.
    """

    def __init__(self, model=None):
        self.model = model

        if self.model:
            self.k = self.model.weights_.shape[0]
            self.covariance = self.model.covariance_type
        else:
            self.k = 0
            self.covariance = None

    def aic(self, X):
        if self.model:
            return self.model.aic(X)
        else:
            return float('inf')

    def bic(self, X):
        if self.model:
            return self.model.bic(X)
        else:
            return float('inf')


def best_gmm(X,
             max_range=np.arange(2, 11),
             covariance_types=None,
             max_iter=1000,
             n_init=5,
             seed=SEED):
    """
    Return the best Gaussian Mixture Model given the data, a range of K values, and two K selection criteria.

    :param X: usage matrix (made of usage vectors)
    :param max_range: range within the number of clusters should lie
    :param covariance_types: a list containing any subset of this list:
    :param max_iter: maximum number of EM iterations
    :param n_init: number of EM runs
    :param seed: random seed
    :return: best GMM according to Akaike Information Criterion, Bayesian Information Criterion,
             and the respective AIC and BIC scores
    """
    if covariance_types is None:
        covariance_types = ['full', 'spherical', 'tied', 'diag']
    if not isinstance(covariance_types, (list,)):
        covariance_types = [covariance_types]

    aics = defaultdict(list)
    bics = defaultdict(list)
    best_gmm_aic = GMM()
    best_gmm_bic = GMM()

    for i, cov in enumerate(covariance_types):
        for k in max_range:
            m = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                max_iter=max_iter,
                n_init=n_init,
                random_state=seed).fit(X)

            if m.aic(X) < best_gmm_aic.aic(X):
                best_gmm_aic = GMM(m)
            if m.bic(X) < best_gmm_bic.bic(X):
                best_gmm_bic = GMM(m)

            bics[cov].append(m.bic(X))
            aics[cov].append(m.aic(X))

    return best_gmm_aic, best_gmm_bic, bics, aics


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
            if mode == 'sum':
                vec = np.sum(vec.reshape((ndims, -1)), axis=1)
            usages_out[w] = (
                np.row_stack((usages_out[w][0], vec)),
                usages_out[w][1] + [context],
                usages_out[w][2] + [pos_in_context],
                usages_out[w][3] + [decade]
            )

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
        best_model, _ = best_kmeans(X, k_range, criterion=criterion)
    elif method == 'gmm':
        best_model_aic, best_model_bic, _, _ = best_gmm(X, k_range)
        if criterion == 'aic':
            best_model = best_model_aic
        elif criterion == 'bic':
            best_model = best_model_bic
        else:
            raise ValueError('Invalid criterion {}. Choose "aic" or "bic".'.format(criterion))
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
    for w in tqdm(usages):
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


def parse_snippet(snippet, tokenizer, target_position):
    """
    Parse a list of wordpiece token ids into a human-readable sentence string.

    :param snippet: list of token ids
    :param tokenizer: BertTokenizer object
    :param target_position: position of the target word in the token list
    :return: sentence string with highlighted target word and reassebled word pieces
    """
    sentence = ''
    tokens = tokenizer.convert_ids_to_tokens(snippet)

    for pos, token in enumerate(tokens):
        if not token.startswith('##'):
            sentence += ' '
        # if token not in [',', '.', ';', ':', '!', '?']:
        #     sentence += ' '

        if pos == target_position:
            sentence += '[[{}]]'.format(token)
        else:
            if token.startswith('##'):
                sentence += token[2:]
            elif token == '[PAD]':
                continue
            elif token == '[UNK]':
                sentence += 'UNK'
            else:
                sentence += token

    return sentence.strip()


def prepare_snippets(usages, clusterings, pretrained_weights='models/bert-base-uncased', bin2label=None):
    """
    Collect usage snippets and organise them according to their usage type and time interval.

    :param usages: dictionary mapping lemmas to their tensor data and metadata
    :param clusterings: dictionary mapping lemmas to their best clustering model
    :param pretrained_weights: path to BERT model folder with weights and config file
    :param bin2label: dictionary mapping time interval to time label
    :return: dictionary mapping (lemma, cluster, time) triplets to lists of usage snippets
    """
    snippets = {}  # (lemma, cluster_id, time_interval) -> [(<s>, ..., <\s>), (<s>, ..., <\s>), ...]
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

    for word in tqdm(usages):
        snippets[word] = defaultdict(lambda: defaultdict(list))

        _, contexts, positions, t_labels = usages[word]
        cl_labels = clusterings[word].labels_

        for context, pos, cl, t in zip(contexts, positions, cl_labels, t_labels):
            if bin2label:
                t = bin2label[t]
            snippets[word][cl][t].append(parse_snippet(context, tokenizer, pos))

    return snippets


def sample_snippets(snippets, time_periods):
    """
    Sample usage examples for each word of interest according to the following procedure.
    For each cluster, uniformly sample a usage snippet for every time period where that cluster appears.
    If the cluster does not appear in a certain time period, sample an alternative time period uniformly.

    :param snippets: dictionary mapping (lemma, cluster, time) triplets to lists of usage snippets
    :param time_periods: range of time periods of interest
    :return: dictionary mapping lemmas to their usage examples as well as to the respective
             cluster id and time interval labels.
    """
    snippet_lists = defaultdict(list)
    snippet_labels = defaultdict(list)

    for w in tqdm(snippets):
        for cl in snippets[w]:
            for t in time_periods:
                population = snippets[w][cl][t]
                sample = None

                while (sample is None) or (sample in snippet_lists[w]):
                    if population:
                        sample_idx = np.random.choice(np.arange(len(population)))
                        sample = population[sample_idx]
                    else:
                        # if there are no snippets of cluster `cl` in time `t`, uniformly sample
                        # an alternative time interval to draw a usage example from
                        sample_t = np.random.choice(list(snippets[w][cl].keys()))
                        population = snippets[w][cl][sample_t]

                snippet_lists[w].append(sample)
                snippet_labels[w].append((cl, t))

    sampled = defaultdict(list)
    for w in snippet_lists:
        for s_idx, (s, (cl, t)) in enumerate(zip(snippet_lists[w], snippet_labels[w])):
            sampled[w].append((w, s_idx, cl, t, s))

    return sampled


def make_usage_pairs(snippets):
    """
    Obtain all possible usage pairs without repetitions from list of usages.

    :param snippets: dictionary mapping lemmas to lists of annotation tuples (snippet_id, cluster_id, time, snippet)
    :return: dictionary mapping lemmas to lists of annotated usage pairs
    """
    usage_pairs = defaultdict(list)
    n = 0
    # obtain all possible pairs without repetitions
    for w in snippets:
        for pair in itertools.combinations(snippets[w], 2):
            usage_pairs[w].append(pair)
            n += 1
    # randomly shuffle order of pairs
    for w in snippets:
        random.shuffle(usage_pairs[w])

    print('{} usage pairs generated.'.format(n))
    return usage_pairs


def make_test_usage_pairs(snippets, shift=True, n_pairs_per_usage=1, max_offset=10):
    """
    Obtain test usage pairs.

    :param snippets: dictionary mapping lemmas to lists of annotation tuples (snippet_id, cluster_id, time, snippet)
    :param shift: whether to shift the sentence by a few tokens (max 10).
    :param n_pairs_per_usage: number of test pairs to generate for each usage example
    :return: dictionary mapping lemmas to lists of annotated test usage pairs
    """
    if n_pairs_per_usage > 1 and not shift:
        raise ValueError(
            'Generating more than 1 pair per usage without shift will result in {} identical pairs for each usage.'.format(
                n_pairs_per_usage))
    if n_pairs_per_usage > max_offset:
        raise ValueError('Can only generate max_offset={} different pairs per usage.'.format(max_offset))

    usage_pairs = defaultdict(list)
    n = 0
    # obtain all test pairs (s, s)
    for w in snippets:
        for snip in snippets[w]:
            i = 0
            offset_range = list(np.arange(max_offset + 1))

            while i < n_pairs_per_usage:
                if shift:
                    (w, s_idx, cl, t, s) = snip

                    # randomly decide how to shift the sentence
                    offset = np.random.choice(offset_range)
                    bos = np.random.choice([0, 1])  # 1: beginning of sentence  0: end of sentence

                    s_shifted = s.split(' ')

                    # shift sentence by `offset` positions starting from beginning or end of sentence
                    n_del = int(offset)
                    if bos:
                        while n_del > 0 and not (s_shifted[0].startswith('[[') and s_shifted[0].endswith(']]')):
                            s_shifted = s_shifted[1:]
                            n_del -= 1
                    else:
                        while n_del > 0 and not (s_shifted[-1].startswith('[[') and s_shifted[-1].endswith(']]')):
                            s_shifted = s_shifted[:-1]
                            n_del -= 1

                    s_shifted = ' '.join(s_shifted)
                    snip2 = (w, s_idx, cl, t, s_shifted, offset, bos)
                else:
                    snip2 = snip

                if random.random() < 0.5:
                    usage_pairs[w].append((snip, snip2))
                else:
                    usage_pairs[w].append((snip2, snip))

                offset_range.remove(offset)
                i += 1
            n += 1

    # randomly shuffle order of pairs
    for w in snippets:
        random.shuffle(usage_pairs[w])

    print('{} usage pairs generated.'.format(n))
    return usage_pairs


def usage_pairs_totsv(usage_pairs, output_path):
    """
    Store all usage pairs in a tsv file.

    :param usage_pairs: dictionary mapping lemmas to lists of annotated usage pairs
    :param output_path: path of the output tsv file containing all usage pair examples
    """
    all_pairs = []
    for w in usage_pairs:
        all_pairs.extend(usage_pairs[w])
    random.shuffle(all_pairs)

    with open(output_path, 'w') as f:
        print('\t'.join(FIGURE_EIGHT_FIELDS), file=f)
        for u1, u2 in all_pairs:
            (w1, s_idx_1, cl_1, t_1, s_1) = u1
            (w2, s_idx_2, cl_2, t_2, s_2) = u2
            assert w1 == w2
            print('\t'.join(map(str, [w1, s_idx_1, s_idx_2, cl_1, cl_2, t_1, t_2, s_1, s_2])), file=f)

    print('Saved to: {}'.format(output_path))


def test_usage_pairs_totsv(usage_pairs, output_path):
    """
    Store all usage pairs in a tsv file.

    :param usage_pairs: dictionary mapping lemmas to lists of annotated usage pairs
    :param output_path: path of the output tsv file containing all usage pair examples
    """
    all_pairs = []
    for w in usage_pairs:
        all_pairs.extend(usage_pairs[w])
    random.shuffle(all_pairs)

    with open(output_path, 'w') as f:
        print('\t'.join(FIGURE_EIGHT_FIELDS + ['offset', 'bos'] + FIGURE_EIGHT_FIELDS_TEST), file=f)
        for u1, u2 in all_pairs:
            (w1, s_idx_1, cl_1, t_1, s_1) = u1
            (w2, s_idx_2, cl_2, t_2, s_2, offset, bos) = u2
            assert w1 == w2
            print('\t'.join(
                map(str, [w1, s_idx_1, s_idx_2, cl_1, cl_2, t_1, t_2, s_1, s_2, offset, bos, 'identical', '', 'TRUE'])),
                  file=f)

    print('Saved to: {}'.format(output_path))


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
