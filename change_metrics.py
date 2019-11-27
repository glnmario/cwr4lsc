import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn import preprocessing
from tqdm import tqdm


def entropy_timeseries(usage_distribution, intervals=None):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: array of entropy values, one for each usage distribution
    """
    if intervals:
        usage_distribution = usage_distribution[:, intervals]

    usage_distribution = preprocessing.normalize(usage_distribution, norm='l1', axis=0)

    H = []
    for t in range(usage_distribution.shape[1]):
        c = usage_distribution[:, t]
        if any(c):
            h = entropy(c)
        else:
            continue  # h = 0.
        H.append(h)

    return np.array(H)


def entropy_difference_timeseries(usage_distribution, absolute=True, intervals=None):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: array of entropy differences between contiguous usage distributions
    """
    if absolute:
        return np.array([abs(d) for d in np.diff(entropy_timeseries(usage_distribution, intervals))])
    else:
        return np.diff(entropy_timeseries(usage_distribution, intervals))


def js_divergence(*usage_distribution):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: Jensen-Shannon Divergence between multiple usage distributions
    """
    clusters = np.vstack(usage_distribution)
    n = clusters.shape[1]
    entropy_of_sum = entropy(1 / n * np.sum(clusters, axis=1))
    sum_of_entropies = 1 / n * np.sum([entropy(clusters[:, t]) for t in range(n)])
    return entropy_of_sum - sum_of_entropies


def js_distance(*usage_distribution):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :return: Jensen-Shannon Distance between two usage distributions
    """
    return np.sqrt(js_divergence(usage_distribution))


def jsd_timeseries(usage_distribution, dfunction=js_divergence, intervals=None):
    """
    :param usage_distribution: a CxT diachronic usage distribution matrix
    :param dfunction: a JSD function (js_divergence or js_distance)
    :return: array of JSD between contiguous usage distributions
    """
    if intervals:
        usage_distribution = usage_distribution[:, intervals]

    usage_distribution = preprocessing.normalize(usage_distribution, norm='l1', axis=0)
    distances = []
    for t in range(usage_distribution.shape[1] - 1):
        c = usage_distribution[:, t]
        c_next = usage_distribution[:, t + 1]

        if any(c) and any(c_next):
            d = dfunction(c_next, c)
        else:
            continue  # d = 0.
        distances.append(d)

    return np.array(distances)


def avg_pairwise_distance_timeseries(usages, metrics=('cosine', 'canberra'), interval_labels=None):
    """
    :param usages: 4-place data structure containing usage vectors, sentences, sentence positions, and time labels
    :param metrics: a list or tuple of metric names for scipy.spatial.distance.cdist
    :return: array of average pairwise distances between contiguous usage distributions
    """
    U_w, contexts, pos_in_context, t_labels = usages

    by_time = {}
    for u_w, sent, pos, t_label in zip(U_w, contexts, pos_in_context, t_labels):
        if interval_labels and t_label not in interval_labels:
            continue
        try:
            by_time[t_label] = np.vstack((by_time[t_label], u_w))
        except KeyError:
            by_time[t_label] = u_w

    sorted_t_labels = sorted(by_time.keys())
    distances = {metric: [] for metric in metrics}
    for i in np.arange(len(sorted_t_labels) - 1):
        t = sorted_t_labels[i]
        t_ = sorted_t_labels[i + 1]
        for metric in metrics:
            distances[metric].append(np.mean(cdist(by_time[t], by_time[t_], metric=metric)))
    distances = {metric: np.array(dists) for metric, dists in distances.items()}

    return distances
