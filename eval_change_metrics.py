import csv
import pickle
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import pandas as pd
from change_metrics import jsd_timeseries, avg_pairwise_distance_timeseries, entropy_difference_timeseries
from clustering import usage_distribution


def gulordava_baroni_correlation(clusterings, usages, word2shift):
    """
    :param clusterings: dictionary mapping target words to their best clustering model
    :param usages: dictionary mapping target words to a 4-place data structure containing
                   usage vectors, sentences, sentence positions, and time labels
    :param word2shift: dictionary mapping target words to human-rated shift scores
    :return: dataframe containing measurements for all JSD, entropy difference, and average pairwise distance
    """
    jsd = []
    dh = []
    avgcos = []
    avgcan = []
    shift = []

    for word in tqdm(clusterings):
        _, _, _, t_labels = usages[word]
        clustering = clusterings[word]

        # create usage distribution based on clustering results
        usage_distr = usage_distribution(clustering.labels_, t_labels)
        usage_distr = preprocessing.normalize(usage_distr, norm='l1', axis=0)

        print('jsd')
        jsd.append(np.mean(jsd_timeseries(usage_distr)))
        print('dh')
        dh.append(np.mean(entropy_difference_timeseries(usage_distr)))
        print('avg d')
        avgd = avg_pairwise_distance_timeseries(usages[word], metrics=['cosine', 'canberra'])
        avgcos.append(np.mean(avgd['cosine']))
        avgcan.append(np.mean(avgd['canberra']))
        shift.append(word2shift[word])

    df = pd.DataFrame(
        data={
            'JSD': jsd,
            'H diff': dh,
            'cosine': avgcos,
            'canberra': avgcan,
            'shift': shift
        },
        index=list(clusterings.keys())
    )

    return df


if __name__ == '__main__':
    # Load usages
    # with open('coha/6090/sum/usages_len128_silhouette.dict', 'rb') as f:
    with open('data/sum/usages_5_len128_all.dict', 'rb') as f:
        usages = pickle.load(f)

    # Load clusterings
    # with open('coha/6090/sum/usages_len128_silhouette.clustering.dict', 'rb') as f:
    with open('data/sum/usages_5_len128_silhouette.clustering.dict', 'rb') as f:
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
