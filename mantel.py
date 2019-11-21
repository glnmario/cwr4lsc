import pickle
from skbio.stats.distance import mantel
import numpy as np

with open('data/GradedMeaningAnnotation/usim_matrices.dict', 'rb') as f:
    sim_matrices = pickle.load(f)

with open('data/GradedMeaningAnnotation/bert_base_matrices_sum.dict', 'rb') as f:
    bert_sim_matrices = pickle.load(f)

coeffs = []
sig_coeffs = []

for w in sim_matrices:
    coeff, p_value, n = mantel(
        sim_matrices[w],
        bert_sim_matrices[w],
        method='spearman',  # pearson
        permutations=999,
        alternative='two-sided'  # greater, less
    )

    print(w)
    print('spearman: {:.2f}    p: {:.2f}'.format(coeff, p_value))

    coeffs.append(coeff)

    if p_value < 0.05:
        sig_coeffs.append(coeff)

print(np.mean(coeffs))
print(np.mean(sig_coeffs))
