from typing import Dict, Any, List, Union
import itertools

def get_count() -> int:
    """Returns the number of trials to run in the sweep."""
    return 2  # specify the number of trials here

sweep_config: Dict[str, Any] = {
    'method': 'random',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'min': 32,
            'max': 256,
            'q': 8
        },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
        },
        'epochs': {
            'values': [10, 20, 30]
        },
        'fc_layer_size': {
            'values': [128, 256, 512]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.1
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        }
    }
}