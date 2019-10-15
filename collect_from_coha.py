import pickle
import argparse
import numpy as np
from usage_collector import collect_from_coha

parser = argparse.ArgumentParser()
parser.add_argument('--seqlen', type=int, default=128)
parser.add_argument('--bertdir', type=str, default='models/bert-base-uncased')
parser.add_argument('--cohadir', type=str, default='data/coha')
parser.add_argument('--outdir', type=str, default='data')
parser.add_argument('--buffer', type=int, default=1024)

args = parser.parse_args()

targets = ['net', 'virtual', 'disk', 'card', 'optical', 'virus',
           'signal', 'mirror', 'energy', 'compact', 'leaf',
           'brick', 'federal', 'sphere', 'coach', 'spine']

print('{}\nSEQUENCE LENGTH: {}\n{}'.format('-' * 30, args.seqlen, '-' * 30))

# decades = list(np.arange(1910, 2001, 10))
# decades = list(np.arange(1810, 1811, 10))

for decade in np.arange(1910, 2009, 10):
    collect_from_coha(targets,
                      [decade],
                      sequence_length=args.seqlen,
                      pretrained_weights=args.bertdir,
                      coha_dir=args.cohadir,
                      output_path='{}/concat/usages_16_len{}_{}.dict'.format(args.outdir, args.seqlen, decade),
                      buffer_size=args.buffer)

    # # Save usages
    # with open('{}/concat/usages_16_len{}_{}.dict'.format(args.outdir, args.seqlen, decade), 'wb') as f:
    #     pickle.dump(usages, file=f)
    # usages = None
