from typing import Dict, Any, List, Union
import itertools

def get_count() -> int:
    """Returns the number of trials to run in the sweep."""
    return 100  # specify the number of trials here


sweep_config: Dict[str, Any] = {
    'method': 'bayes',
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'epochs': {
            'values': [50]
        },
        'scheduler': {
            'values': ['StepLR'],
            'distribution': 'categorical'
        },
        'scheduler_type': {
            'values': ['StepLR'],
            'distribution': 'categorical'
        },
        'gamma': {
            'min': 0.05,
            'max': 0.2,
            'distribution': 'uniform'
        },
        'early_stopping_patience': {
            'min': 3,
            'max': 10,
            'distribution': 'int_uniform'
        },
        'early_stopping_delta': {
            'min': 0.00005,
            'max': 0.0002,
            'distribution': 'uniform'
        },
        'learning_rate': {
            'min': 0.0005,
            'max': 0.002,
            'distribution': 'uniform'
        },
        'optimizer': {
            'values': ['Adam'],
            'distribution': 'categorical'
        },
        'weight_decay': {
            'min': 0.00005,
            'max': 0.0002,
            'distribution': 'uniform'
        },
        'batch_size': {
            'values': [32, 64, 128],
            'distribution': 'categorical'
        },
        'step_size': {
            'min': 5,
            'max': 20,
            'distribution': 'int_uniform'
        }
    }
}


'''
# working basic params
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
'''