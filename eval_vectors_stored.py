from operator import itemgetter
import os
import numpy as np
import csv
from collections import defaultdict
import pickle
from tqdm import tqdm
import pickle
import itertools
from skbio.stats.distance import mantel
from scipy.stats import spearmanr
import plotly.graph_objects as go


print('Load average annotation variance per lemma')
mean_var = {}
with open('figure8/pilot/mean_var.txt', 'r') as f:
    for line in f.readlines():
        split = line.split()
        mean_var[split[0]] = float(split[1])

print('Load average annotation entropy per lemma')
mean_entropy = {}
with open('figure8/pilot/mean_entropy.txt', 'r') as f:
    for line in f.readlines():
        split = line.split()
        mean_entropy[split[0]] = float(split[1])

print('Load shift scores')
shift_scores = {}
with open('data/shift_scores.txt', 'r') as f:
    for line in f.readlines():
        word, score = line.split()
        shift_scores[word] = float(score)

print('Load human judgements')
with open('figure8/human-sim-matrices.dict', 'rb') as f:
    sim_matrices = pickle.load(f)

dir_path = 'eval_results'
n_significant = dict()
avg_significant_corr = dict()

# for MODE in ['sum', 'cat']:
for entry in os.scandir(dir_path):
    try:
        model, file_type = entry.path.split('.')
        model = model.split('/')[1]
        layers = model.split('-')[-1]
    except ValueError:
        continue  # ends with .corrs.dict

    if file_type != 'dict':
        continue

    if len(layers.split(',')) > 1:
        continue

    print('\n>>', model)

    # print('Load model judgements: {}'.format(MODE))
    # with open('bert-large-uncased-sum-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24.dict', 'rb') as f:
    # with open(entry.path, 'rb') as f:
    with open('eval_results/bert-base-uncased-sum-1,2,3,4,5,6,7,8,9,10,11,12.dict', 'rb') as f:
        bert_sim_matrices = pickle.load(f)

    print('Mantel test')

    coeffs = {}
    sig_coeffs = {}
    unsig_coeffs = {}
    for w in bert_sim_matrices:
        coeff, p_value, n = mantel(
            sim_matrices[w],
            bert_sim_matrices[w],
            method='spearman',  # pearson
            permutations=999,
            alternative='two-sided'  # greater, less
        )

        coeffs[w] = coeff, p_value
        if p_value < 0.05:
            sig_coeffs[w] = coeff, p_value
        else:
            unsig_coeffs[w] = coeff, p_value

    print('{}/{} significant correlations.'.format(len(sig_coeffs), len(coeffs)))

    n_significant[layers] = len(sig_coeffs)
    avg_significant_corr[layers] = np.mean([r for r, p in sig_coeffs.values()])

    printout_triples = []
    for w in sig_coeffs:
        coeff, p_value = sig_coeffs[w]
        printout_triples.append((w, coeff, p_value))

    for (w, coeff, p_value) in sorted(printout_triples, key=itemgetter(1)):
        print('{:10} & {:.3f} & {:.3f} \\\\'.format(w, coeff, p_value))
        # print('{:10} r: {:-.3f}    p: {:.3f}'.format(w, coeff, p_value))

    print('\\hline')

    printout_triples = []
    for w in unsig_coeffs:
        coeff, p_value = unsig_coeffs[w]
        printout_triples.append((w, coeff, p_value))

    for (w, coeff, p_value) in sorted(printout_triples, key=itemgetter(1)):
        print('{:10} & {:.3f} & {:.3f} \\\\'.format(w, coeff, p_value))
        # print('{:10} r: {:-.3f}    p: {:.3f}'.format(w, coeff, p_value))


    eval_coeffs = [coeffs[w][0] for w in coeffs]
    mean_var_list = [mean_var[w] for w in coeffs]
    mean_entropy_list = [mean_entropy[w] for w in mean_var]
    print('\nCorrelation between evaluation results and disagreement.')
    print('Correlation with mean variance')
    r, p = spearmanr(eval_coeffs, mean_var_list)
    end = '    !!!\n' if p < 0.05 else '\n'
    print('r: {:-.3f}    p: {:.2f}'.format(r, p))
    print('Correlation with mean entropy')
    r, p = spearmanr(eval_coeffs, mean_entropy_list)
    end = '    !!!\n' if p < 0.05 else '\n'
    print('r: {:-.3f}    p: {:.2f}'.format(r, p))

    eval_coeffs = [coeffs[w][0] for w in unsig_coeffs]
    mean_var_list = [mean_var[w] for w in unsig_coeffs]
    mean_entropy_list = [mean_entropy[w] for w in unsig_coeffs]
    print('\nCorrelation between non-significant evaluation results and disagreement.')
    print('Correlation with mean variance')
    r, p = spearmanr(eval_coeffs, mean_var_list)
    end = '    !!!\n' if p < 0.05 else '\n'
    print('r: {:-.3f}    p: {:.2f}'.format(r, p))
    print('Correlation with mean entropy')
    r, p = spearmanr(eval_coeffs, mean_entropy_list)
    end = '    !!!\n' if p < 0.05 else '\n'
    print('r: {:-.3f}    p: {:.2f}'.format(r, p))


    eval_coeffs = [coeffs[w][0] for w in coeffs]
    shift_score_list = [[w] for w in coeffs]
    print('\nCorrelation between evaluation results and shift scores.')
    r, p = spearmanr(eval_coeffs, shift_score_list)
    end = '    !!!\n' if p < 0.05 else '\n'
    print('r: {:-.3f}    p: {:.2f}'.format(r, p))

    eval_coeffs = [coeffs[w][0] for w in unsig_coeffs]
    shift_score_list = [shift_scores[w] for w in unsig_coeffs]
    print('Correlation between non-significant evaluation results and shift scores.')
    r, p = spearmanr(eval_coeffs, shift_score_list)
    end = '    !!!\n' if p < 0.05 else '\n'
    print('r: {:-.3f}    p: {:.2f}'.format(r, p))


# print('\n\n\nlayers, n-sign, avg-sign-corr')
# for layers in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
#                '1,2', '2,3', '3,4', '4,5', '5,6', '6,7', '7,8', '8,9', '9,10', '10,11', '11,12']:
#     print('{:6} {:5}\t{:.3}'.format(layers, n_significant[layers], avg_significant_corr[layers]))
