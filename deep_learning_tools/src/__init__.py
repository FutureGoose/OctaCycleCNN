"""Deep Learning Tools - A library for training and evaluating deep learning models."""

from .training import ModelTrainer, EarlyStopping, accuracy, precision, recall, f1_score
from .visualization import MetricsPlotter
from .logging import (
    LoggerManager, BaseLogger, create_logger,
    NullLogger, FileLogger, TensorBoardLogger, WandBLogger
)
from .utils import (
    prepare_datasets, KarpathyVerification, 
    AlternatingFlipDataset, MultiCropTTAWrapper,
    PrintManager
)
from .sweeps import run_sweep, sweep_config, get_count

__version__ = '0.1.0'

__all__ = [
    # Training
    'ModelTrainer',
    'EarlyStopping',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    # Visualization
    'MetricsPlotter',
    # Logging
    'LoggerManager',
    'BaseLogger',
    'create_logger',
    'NullLogger',
    'FileLogger',
    'TensorBoardLogger',
    'WandBLogger',
    # Utils
    'prepare_datasets',
    'KarpathyVerification',
    'AlternatingFlipDataset',
    'MultiCropTTAWrapper',
    'PrintManager',
    # Sweeps
    'run_sweep',
    'sweep_config',
    'get_count',
]
