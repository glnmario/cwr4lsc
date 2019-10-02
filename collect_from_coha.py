import pickle
import numpy as np
from usage_collector import collect_from_coha


# Load target words
targets = []
with open('data/gulordava-baroni-binmin10.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line:
            targets.append(line)

# targets = ['common', 'sense']


decades = list(np.arange(1910, 2001, 10))
# decades = list(np.arange(1810, 1811, 10))

usages = collect_from_coha(targets,
                           decades,
                           pretrained_weights='models/bert-base-uncased',
                           buffer_size=1024)

# Save usages
with open('data/usages.dict', 'wb') as f:
    pickle.dump(usages, file=f)
