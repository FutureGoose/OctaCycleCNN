sweep_config = {
    'method': 'random',  # search method
    'metric': {
        'name': 'val/loss',  # metric to optimize
        'goal': 'minimize'    # whether to minimize or maximize
    },
    'parameters': {
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'fc_layer_size': {
            'values': [128, 256, 512]
        },
        'dropout': {
            'values': [0.3, 0.4, 0.5]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.01
        },
        'batch_size': {
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 32,
            'max': 256
        },
        'epochs': {
            'value': 10  # fixed value, not varying
        }
    }
}