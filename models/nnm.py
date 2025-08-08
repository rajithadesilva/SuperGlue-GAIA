import torch
from torch import nn

from models.hloc.matchers.nearest_neighbor import find_nn, mutual_check

CONFS = {
    'NNM': {
        'output': 'NNM',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },
    'ONN': {
        'output': 'ONN',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': False,
            'distance_threshold': None,
        },
    },
    'NNR': {
        'output': 'NNR',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.9,
        },
    }
}

class NNM(nn.Module):
    def __init__(self, config=CONFS['ONN']['model']):
        super().__init__()
        self.config = {
            'ratio_threshold': config.get('ratio_threshold', None),
            'distance_threshold': config.get('distance_threshold', None),
            'match_threshold': 0.0, #Dummy value
            'do_mutual_check': config.get('do_mutual_check', True),
        }

    def forward(self, data):
        desc0 = data['descriptors0']  # shape: [B, D, N]
        desc1 = data['descriptors1']  # shape: [B, D, M]

        sim = torch.einsum('bdn,bdm->bnm', desc0, desc1)
        matches0, scores0 = find_nn(
            sim, 
            self.config['ratio_threshold'], 
            self.config['distance_threshold']
        )
        matches1, scores1 = find_nn(
            sim.transpose(1, 2), 
            self.config['ratio_threshold'], 
            self.config['distance_threshold']
        )

        if self.config['do_mutual_check']:
            matches0 = mutual_check(matches0, matches1)
            matches1 = mutual_check(matches1, matches0)

        return {
            'matches0': matches0,  # shape: [B, N]
            'matches1': matches1,  # shape: [B, M]
            'matching_scores0': scores0,  # shape: [B, N]
            'matching_scores1': scores1,  # shape: [B, M]
        }
