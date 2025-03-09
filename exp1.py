# calculate LogME, NCE and LEEP with cached features and outputs

import os
import argparse
from engines.assess import assess_with_cache

def parser_args():
    parser = argparse.ArgumentParser(description='assess SNN with cache')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory of cache')
    parser.add_argument('--dataset', default='n_mnist', type=str, help='dataset')
    parser.add_argument('--dataset_type', default='train', type=str, help='dataset type')
    return parser.parse_args()


model_names = [
    'spiking_mlp12',
    'att_snn',
    'sew_resnet18',
    'sew_resnet34',
    'sew_resnet50',
    'sew_resnet101',
    'sew_resnet152',
    'spiking_resnet18',
    'spiking_resnet34',
    'spiking_resnet50',
]

if __name__ == '__main__':
    args = parser_args()
    for model in model_names:
        cache_dir = os.path.join(args.output_dir, 'cache', args.dataset, model, args.dataset_type)
        if os.path.exists(cache_dir):
            score1, score2, score3 = assess_with_cache(cache_dir=cache_dir)
            print('NCE: {:.8f}, LEEP: {:.8f}, logME: {:.8f}, model: {}'.format(score1, score2, score3, model))