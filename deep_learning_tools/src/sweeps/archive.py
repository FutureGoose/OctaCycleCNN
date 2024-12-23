'''
# sweep.py ver for tuning cnn and nn architecture
def train_function(trainer: "ModelTrainer", config: Dict[str, Any]):
    """training function that will be called by the sweep."""
    try:
        # create new model instance with config parameters
        model = EightLayerConvNet(
            num_classes=10,
            conv1_channels=config["conv1_channels"],
            conv2_channels=config["conv2_channels"],
            conv3_channels=config["conv3_channels"],
            fc_layer_size=config["fc_layer_size"]
        ).to(trainer.device)
        
        # update trainer with new model and config parameters
        trainer.model = model
        trainer.batch_size = config["batch_size"]
        trainer.optimizer = build_optimizer(
            network=trainer.model,
            optimizer_name=config["optimizer"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        # configure scheduler if specified
        if config["scheduler"] == "StepLR":
            trainer.scheduler = StepLR(
                trainer.optimizer, 
                step_size=config["step_size"], 
                gamma=config["gamma"]
            )

        # update early stopping parameters
        trainer.early_stopping.patience = config["early_stopping_patience"]
        trainer.early_stopping.delta = config["early_stopping_delta"]

        # set up data loaders
        trainer.setup_data_loaders(training_set=trainer.training_set, val_set=trainer.val_set)

        # train for the specified number of epochs
        for epoch in range(1, config.get("epochs", 50) + 1):
            trainer.metrics_history['epochs'].append(epoch)
            train_loss = sweep_train_epoch(trainer, epoch)
            val_loss = sweep_evaluate(trainer, epoch, phase='val')

            if trainer.scheduler:
                trainer.scheduler.step()

            if trainer.early_stopping.early_stop:
                print(f"\nearly stopping triggered at epoch {epoch}")
                break

    except Exception as e:
        print(f"error during training: {str(e)}")
        traceback.print_exc()
        raise  # re-raise the exception to ensure the sweep notices the failure
'''



'''
sweep_config.py for tuning cnn and nn architecture
# extra params for tuning cnn and nn architecture
        'conv1_channels': {
            'values': [32, 64, 128],
            'distribution': 'categorical'
        },
        'conv2_channels': {
            'values': [64, 128, 256],
            'distribution': 'categorical'
        },
        'conv3_channels': {
            'values': [128, 256, 512],
            'distribution': 'categorical'
        },
        'fc_layer_size': {
            'values': [256, 512, 1024],
            'distribution': 'categorical'
        },
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