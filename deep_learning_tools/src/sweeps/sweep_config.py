from typing import Dict, Any, List, Union
import itertools

def get_count() -> int:
    """Returns the number of trials to run in the sweep."""
    return 2  # Specify the number of trials here

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
        # Made 'epochs' a hyperparameter instead of fixed value
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

# temp disable below, let be
# # validate sweep configuration
# def validate_sweep_config(config: Dict[str, Any]) -> None:
#     """Validate the sweep configuration."""
#     # check required fields
#     required_fields = ['method', 'metric', 'parameters']
#     for field in required_fields:
#         if field not in config:
#             raise ValueError(f"Missing required field: {field}")
    
#     # check metric configuration
#     if 'name' not in config['metric'] or 'goal' not in config['metric']:
#         raise ValueError("Metric must specify 'name' and 'goal'")
    
#     if config['metric']['goal'] not in ['minimize', 'maximize']:
#         raise ValueError("Metric goal must be 'minimize' or 'maximize'")
    
#     # validate parameters
#     if not config['parameters']:
#         raise ValueError("No parameters specified for sweep")
    
#     # calculate total possible combinations for grid search
#     if config['method'] == 'grid':
#         combinations = 1
#         for param in config['parameters'].values():
#             if 'values' in param:
#                 combinations *= len(param['values'])
#             elif all(k in param for k in ['min', 'max', 'q']):
#                 steps = (param['max'] - param['min']) / param['q']
#                 combinations *= int(steps) + 1
        
#         # warn if count exceeds combinations
#         count = config.get('count', 20)
#         if count > combinations:
#             print(f"warning: sweep count ({count}) exceeds total possible combinations ({combinations})")
#             print(f"consider reducing count to {combinations} for grid search")

# # validate the configuration
# validate_sweep_config(sweep_config)