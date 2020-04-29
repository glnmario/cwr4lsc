import pickle
from clustering import obtain_clusterings, plot_usage_distribution

with open('data/gulordava/century/usages_len128.dict', 'rb') as f:
    usages = pickle.load(f)

clusterings = obtain_clusterings(
    usages,
    out_path='data/gulordava/century/usages_len128.clustering.2.dict',
    method='kmeans',
    criterion='silhouette'
)

plot_usage_distribution(usages, clusterings, '/Users/mario/Desktop/plots/new/', normalized=True)
