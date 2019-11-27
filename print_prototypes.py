import pickle
from clustering import get_prototypes


with open('data/sum/usages_16_len128_silhouette.clustering.dict', 'rb') as f:
    clusterings = pickle.load(f)

with open('data/sum/usages_16_len128_all.dict', 'rb') as f:
    usages = pickle.load(f)


for w in usages:
    p = get_prototypes(w, clusterings[w], usages[w])
    print('>>>', w.upper())
    for i, cluster in enumerate(p):
        print('cluster', i)
        for s in cluster:
            print(s)
    print()