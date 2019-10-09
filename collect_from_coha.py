import pickle
import numpy as np
from usage_collector import collect_from_coha


# Load target words
# targets = []
# with open('data/gulordava-baroni-binmin10.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.strip()
#         if line:
#             targets.append(line)

targets = ['net', 'virtual', 'disk', 'card', 'optical', 'virus',
           'signal', 'mirror', 'energy', 'compact', 'leaf',
           'brick', 'federal', 'sphere', 'coach', 'spine']
# seq_len = 256

for seq_len in [128, 256]:
    print('{}\nSEQUENCE LENGTH: {}\n{}'.format('-'*30, seq_len, '-'*30))

    decades = list(np.arange(1910, 2001, 10))
    # decades = list(np.arange(1810, 1811, 10))

    usages = collect_from_coha(targets,
                               decades,
                               sequence_length=seq_len,
                               pretrained_weights='models/bert-base-uncased',
                               buffer_size=1024)

    # Save usages
    with open('data/usages_16_len{}.dict'.format(seq_len), 'wb') as f:
        pickle.dump(usages, file=f)
