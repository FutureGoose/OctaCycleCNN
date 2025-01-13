from typing import Dict, Any, List, Union
import itertools

def get_count() -> int:
    """Returns the number of trials to run in the sweep."""
    return 20


sweep_config: Dict[str, Any] = {
    'method': 'bayes',  # Bayesian optimization is more efficient for expensive models
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',  # Changed to specify actual values
            'min': 0.001,  # 0.01 / 10
            'max': 0.03    # 0.01 * 3
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.85,   # slightly below 0.9
            'max': 0.95    # slightly above 0.9
        },
        'weight_decay': {
            'distribution': 'log_uniform_values',  # Changed to specify actual values
            'min': 1e-7,   # 1e-6 / 10
            'max': 1e-5    # 1e-6 * 10
        },
        'batch_size': {
            'values': [32, 64]  # including current 32 and reasonable increases
        },
        'optimizer': {
            'values': ['SGD']  # fixed to SGD as requested
        },
        'epochs': {
            'values': [1]
        }
    }
}


'''
def get_count() -> int:
    """Returns the number of trials to run in the sweep."""
    return 10  # specify the number of trials here


sweep_config: Dict[str, Any] = {
    'method': 'bayes',  # bayesian optimization is better than random for our case
    'metric': {
        'goal': 'minimize',
        'name': 'val_loss'
    },
    'parameters': {
        'learning_rate': {
            'min': 0.0001,
            'max': 0.1,
            'distribution': 'uniform'  # simpler than log_uniform
        },
        'optimizer': {
            'values': ['SGD']
        },
        'batch_size': {
            'values': [64, 128]
        },
        'weight_decay': {
            'min': 0.00001,
            'max': 0.0001,
            'distribution': 'uniform'
        },
        'scheduler': {
            'values': ['StepLR']
        },
        'step_size': {
            'min': 10,
            'max': 30,
            'distribution': 'int_uniform'
        }
    }
}
'''
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