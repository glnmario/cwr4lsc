import pickle
import argparse
import numpy as np
from usage_collector import collect_from_coha

parser = argparse.ArgumentParser()
parser.add_argument('--seqlen', type=list, default=[128, 256])
parser.add_argument('--bertdir', type=str, default='models/bert-base-uncased')
parser.add_argument('--cohadir', type=str, default='data/coha')
parser.add_argument('--outdir', type=str, default='data')
parser.add_argument('--buffer', type=int, default=1024)

args = parser.parse_args()


targets = ['net', 'virtual', 'disk', 'card', 'optical', 'virus',
           'signal', 'mirror', 'energy', 'compact', 'leaf',
           'brick', 'federal', 'sphere', 'coach', 'spine']


for seq_len in args.seqlen:
    print('{}\nSEQUENCE LENGTH: {}\n{}'.format('-'*30, seq_len, '-'*30))

    decades = list(np.arange(1910, 2001, 10))
    # decades = list(np.arange(1810, 1811, 10))

    usages = collect_from_coha(targets,
                               decades,
                               sequence_length=seq_len,
                               pretrained_weights=args.bertdir,
                               coha_dir=args.cohadir,
                               buffer_size=args.buffer)

    # Save usages
    with open('{}/usages_16_len{}.dict'.format(args.outdir, seq_len), 'wb') as f:
        pickle.dump(usages, file=f)
