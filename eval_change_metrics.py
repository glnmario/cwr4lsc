import csv
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import pandas as pd
from change_metrics import jsd_timeseries, avg_pairwise_distance_timeseries, entropy_difference_timeseries, js_distance, \
    js_divergence
from clustering import usage_distribution


def gulordava_baroni_correlation(clusterings, usages, word2shift):
    """
    :param clusterings: dictionary mapping target words to their best clustering model
    :param usages: dictionary mapping target words to a 4-place data structure containing
                   usage vectors, sentences, sentence positions, and time labels
    :param word2shift: dictionary mapping target words to human-rated shift scores
    :return: dataframe containing measurements for all JSD, entropy difference, and average pairwise distance
    """
    jsd_multi = []
    jsd_mean = []
    dh_mean = []
    jsd_max = []
    dh_max = []
    jsd_min = []
    dh_min = []
    jsd_median = []
    dh_median = []
    avgcos_mean = []
    avgcan_mean = []
    avgcos_max = []
    avgcan_max = []
    avgcos_min = []
    avgcan_min = []
    avgcos_median = []
    avgcan_median = []
    avgeuc_mean = []
    avgeuc_max = []
    avgeuc_min = []
    avgeuc_median = []
    shift = []

    for word in tqdm(clusterings):
        _, _, _, t_labels = usages[word]
        clustering = clusterings[word]

        # create usage distribution based on clustering results
        usage_distr = usage_distribution(clustering.labels_, t_labels)
        usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)

        intervals = [5, 6, 7, 8]
        interval_labels = [1960, 1970, 1980, 1990]
        # intervals = [5, 8]
        # interval_labels = [1960, 1990]

        # JSD
        jsd = jsd_timeseries(usage_distr, dfunction=js_divergence, intervals=intervals) / usage_distr.shape[0]
        jsd_multi.append(js_divergence([usage_distr[:, t] for t in intervals]))
        jsd_mean.append(np.mean(jsd))
        jsd_max.append(np.max(jsd))
        jsd_min.append(np.min(jsd))
        jsd_median.append(np.median(jsd))

        # Entropy difference
        dh = entropy_difference_timeseries(usage_distr, absolute=False, intervals=intervals) / usage_distr.shape[0]
        dh_mean.append(np.mean(dh))
        dh_max.append(np.max(dh))
        dh_min.append(np.min(dh))
        dh_median.append(np.median(dh))

        # Average pairwise distance
        avgd = avg_pairwise_distance_timeseries(
            usages[word],
            metrics=['cosine', 'canberra', 'euclidean'],
            interval_labels=interval_labels
        )
        # cosine
        avgcos_mean.append(np.mean(avgd['cosine']))
        avgcos_max.append(np.max(avgd['cosine']))
        avgcos_min.append(np.min(avgd['cosine']))
        avgcos_median.append(np.median(avgd['cosine']))
        # canberra
        avgcan_mean.append(np.mean(avgd['canberra']))
        avgcan_max.append(np.max(avgd['canberra']))
        avgcan_min.append(np.min(avgd['canberra']))
        avgcan_median.append(np.median(avgd['canberra']))
        # euclidean
        avgeuc_mean.append(np.mean(avgd['euclidean']))
        avgeuc_max.append(np.max(avgd['euclidean']))
        avgeuc_min.append(np.min(avgd['euclidean']))
        avgeuc_median.append(np.median(avgd['euclidean']))

        shift.append(word2shift[word])

    df = pd.DataFrame(
        data={
            'JSD mean': jsd_mean,
            'JSD max': jsd_max,
            'JSD min': jsd_min,
            'JSD median': jsd_median,
            'JSD multi': jsd_multi,
            'HD mean': dh_mean,
            'HD max': dh_max,
            'HD min': dh_min,
            'HD median': dh_median,
            'cosine mean': avgcos_mean,
            'cosine max': avgcos_max,
            'cosine min': avgcos_min,
            'cosine median': avgcos_median,
            'canberra mean': avgcan_mean,
            'canberra max': avgcan_max,
            'canberra min': avgcan_min,
            'canberra median': avgcan_median,
            'euclidean mean': avgeuc_mean,
            'euclidean max': avgeuc_max,
            'euclidean min': avgeuc_min,
            'euclidean median': avgeuc_median,
            'shift': shift
        },
        index=list(clusterings.keys())
    )

    return df


if __name__ == '__main__':
    # Load usages
    # with open('coha/6090/sum/usages_len128_silhouette.dict', 'rb') as f:
    with open('data/sum/usages_16_len128_all.dict', 'rb') as f:
        usages = pickle.load(f)

    # Load clusterings
    # with open('coha/6090/sum/usages_len128_silhouette.clustering.dict', 'rb') as f:
    with open('data/sum/usages_16_len128_silhouette.clustering.dict', 'rb') as f:
        clusterings = pickle.load(f)

    # Load shift scores from (Gulordava & Baroni, 2011)
    word2shift = {}
    with open('data/gulordava-baroni-eval.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            w, s = row
            word2shift[w] = float(s)
    assert len(word2shift) == 100

    df = gulordava_baroni_correlation(clusterings, usages, word2shift)

    print(df, '\n')
    print('Spearman correlation matrix')
    print(df.corr('spearman'), '\n')

    for metric in df.columns:
        df = df.sort_values(metric, ascending=False)
        print(df[metric], '\n')