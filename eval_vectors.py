import numpy as np
import csv
from collections import defaultdict
from transformers import BertTokenizer, BertModel
import torch
from skbio.stats.distance import mantel
from operator import itemgetter
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import pickle
import itertools

# Load results aggregated by usage pair - using averaging
all_data = []
with open('figure8/aggregate-avg-all.csv', newline='\n', mode='r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        all_data.append(row)

# field name -> field idx
f2i = {f: i for i, f in enumerate(all_data[0])}

# remove fields row
all_data = all_data[1:]


# Collect usages and judgements
snippets = defaultdict(dict)
judgements = defaultdict(dict)
for datum in all_data:
    lemma = datum[f2i['lemma']]
    id_a = int(datum[f2i['id_a']])
    id_b = int(datum[f2i['id_b']])
    a = datum[f2i['a']]
    b = datum[f2i['b']]

    # store sentence in word-specific snippet list
    if a not in snippets[lemma]:
        snippets[lemma][id_a] = a.lower()
    if b not in snippets[lemma]:
        snippets[lemma][id_b] = b.lower()

    # store judgement in word-specific score list
    judgements[lemma][(id_a, id_b)] = float(datum[f2i['sim_score']])


# Reformat snippets so that it's clear what is the form of the target word (and its position in the sentence)
for w in snippets:
    for id_, sent in snippets[w].items():
        tokens = list(map(str.lower, sent.split()))
        form = None
        for t in tokens:
            if t.startswith('[[') and t.endswith(']]'):
                form = t[2:-2]
        snippets[w][id_] = (form, sent)


sim_matrices = {}
for w in judgements:
    n_sent = len(snippets[w])
    m = np.zeros((n_sent, n_sent))
    for (id_a, id_b), score in judgements[w].items():
        m[id_a, id_b] = float(score)
        m[id_b, id_a] = float(score)
    sim_matrices[w] = m



# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
#
# # lm = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# lm = BertModel.from_pretrained(
#     'bert-large-uncased',
#     output_hidden_states=True)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def canberra_distance(a, b):
    return np.sum(np.abs(a-b) / (np.abs(a) + np.abs(b)))


def integers_contiguous(array):
    for i in range(1, len(array)):
        if array[i] != array[i - 1] + 1:
            return False
    return True


def layer_sequences(min, max, reverse=True):
    range_ = list(range(min, max+1))
    for n_layers in sorted(range(1, max - min + 2), reverse=reverse):
        for seq in itertools.permutations(range_, n_layers):
            seq = list(seq)
            if seq == sorted(seq) and integers_contiguous(seq):
                yield seq


for MODEL in ['bert-base-uncased']:  #, 'bert-large-uncased']:
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    lm = BertModel.from_pretrained(MODEL, output_hidden_states=True)

    LAST_LAYER = 12 if MODEL == 'bert-base-uncased' else 24


    for START_LAYER in [1, 0]:
            # for END_LAYER in range(-6, 0):
        for LAYER_SEQ in layer_sequences(START_LAYER, LAST_LAYER):

            for MODE in ['sum', 'cat']:
                print(MODE, LAYER_SEQ)

                bert_sim_matrices = {}
                for lemma in tqdm(judgements):
                # for lemma in ['virus', 'sphere']:

                    bert_sim_matrices[lemma] = np.zeros_like(sim_matrices[lemma])

                    for (id_a, id_b) in judgements[lemma]:

                        form1, s1 = snippets[lemma][id_a]
                        form2, s2 = snippets[lemma][id_b]

                        tokens_s1 = tokenizer.tokenize(s1)
                        tokens_s2 = tokenizer.tokenize(s2)

                        new_tokens_s1 = []
                        skip_till = -1
                        target1_pos = None
                        for i, tok in enumerate(tokens_s1):
                            if i <= skip_till:
                                continue
                            if tok == '[' and tokens_s1[i + 1] == '[' and tokens_s1[i + 2] == form1:
                                skip_till = i + 4
                                target1_pos = len(new_tokens_s1)
                                new_tokens_s1.append(form1)
                            elif tok == '[' and tokens_s1[i + 1] == '[' and tokens_s1[i + 2] == lemma and tokens_s1[i + 3].startswith(
                                    '##'):
                                skip_till = i + 5
                                target1_pos = len(new_tokens_s1)
                                new_tokens_s1.append(lemma)
                                new_tokens_s1.append(tokens_s1[i + 3])
                            else:
                                new_tokens_s1.append(tok)

                        new_tokens_s2 = []
                        skip_till = -1
                        target2_pos = None
                        for i, tok in enumerate(tokens_s2):
                            if i <= skip_till:
                                continue
                            if tok == '[' and tokens_s2[i + 1] == '[' and tokens_s2[i + 2] == form2:
                                skip_till = i + 4
                                target2_pos = len(new_tokens_s2)
                                new_tokens_s2.append(form2)
                            elif tok == '[' and tokens_s2[i + 1] == '[' and tokens_s2[i + 2] == lemma and tokens_s2[i + 3].startswith(
                                    '##'):
                                skip_till = i + 5
                                target2_pos = len(new_tokens_s2)
                                new_tokens_s2.append(lemma)
                                new_tokens_s2.append(tokens_s2[i + 3])
                            else:
                                new_tokens_s2.append(tok)

                        token_ids_1 = tokenizer.encode(new_tokens_s1)
                        token_ids_2 = tokenizer.encode(new_tokens_s2)

                        with torch.no_grad():
                            input_ids_tensor_1 = torch.tensor([token_ids_1])
                            input_ids_tensor_2 = torch.tensor([token_ids_2])

                            outputs_1 = lm(input_ids_tensor_1)
                            outputs_2 = lm(input_ids_tensor_2)

                            #             print(outputs_1[0].shape, outputs_2[1].shape)

                            hidden_states_1 = np.stack([l.clone().numpy() for l in outputs_1[2]])
                            hidden_states_2 = np.stack([l.clone().numpy() for l in outputs_2[2]])

                            hidden_states_1 = hidden_states_1.squeeze(1)
                            hidden_states_2 = hidden_states_2.squeeze(1)

                            usage_vector_1 = hidden_states_1[LAYER_SEQ, target1_pos, :]
                            usage_vector_2 = hidden_states_2[LAYER_SEQ, target2_pos, :]

                            if MODE == 'sum':
                                usage_vector_1 = np.sum(usage_vector_1, axis=0)
                                usage_vector_2 = np.sum(usage_vector_2, axis=0)
                            else:
                                usage_vector_1 = usage_vector_1.reshape((usage_vector_1.shape[0] * usage_vector_1.shape[1]))
                                usage_vector_2 = usage_vector_2.reshape((usage_vector_2.shape[0] * usage_vector_2.shape[1]))
                            #
                            #             print(usage_vector_1.shape, usage_vector_2.shape)

                            sim_score = cosine_similarity(usage_vector_1, usage_vector_2)
                            bert_sim_matrices[lemma][id_a, id_b] = sim_score
                            bert_sim_matrices[lemma][id_b, id_a] = sim_score

                layer_seq_str = ','.join(list(map(str, LAYER_SEQ)))
                with open('{}-{}-{}.dict'.format(MODEL, MODE, layer_seq_str), 'wb') as f:
                    pickle.dump(obj=bert_sim_matrices, file=f)

                coeffs = {}
                sig_coeffs = {}
                for w in bert_sim_matrices:
                    coeff, p_value, n = mantel(
                        sim_matrices[w],
                        bert_sim_matrices[w],
                        method='spearman',  # pearson
                        permutations=999,
                        alternative='two-sided'  # greater, less
                    )
                    print(w)
                    print('spearman: {:.2f}    p: {:.2f}'.format(coeff, p_value))

                    coeffs[w] = coeff, p_value
                    if p_value < 0.05:
                        sig_coeffs[w] = coeff, p_value

                print('{}/{} significant correlations'.format(len(sig_coeffs), len(coeffs)))
                for w, (c, p) in sig_coeffs:
                    print('{}  spearman: {:.2f}    p: {:.2f}'.format(w, c, p))

                with open('{}-{}-{}.corrs.dict'.format(MODEL, MODE, layer_seq_str), 'wb') as f:
                    pickle.dump(obj=bert_sim_matrices, file=f)


# with open('bert-base-uncased_sum-2-.dict', 'rb') as f:
#     bert_sim_matrices = pickle.load(f)
#
# coeffs = {}
# sig_coeffs = {}
# for w in bert_sim_matrices:
#     coeff, p_value, n = mantel(
#         sim_matrices[w],
#         bert_sim_matrices[w],
#         method='spearman',  # pearson
#         permutations=999,
#         alternative='two-sided'  # greater, less
#     )
#     print(w)
#     print('spearman: {:.2f}    p: {:.2f}'.format(coeff, p_value))
#
#     coeffs[w] = coeff
#     if p_value < 0.05:
#         sig_coeffs[w] = coeff
#         print('---')


