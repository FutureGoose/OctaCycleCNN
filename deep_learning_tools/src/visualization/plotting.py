import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from typing import Optional, List

class MetricsPlotter:
    """handles all plotting functionality for training metrics."""
    
    @staticmethod
    def plot_losses(ax, epochs: List[int], train_losses: List[float], 
                   test_losses: List[float], batch_variation: Optional[tuple] = None):
        """plots training and test losses."""
        # plot main loss lines
        ax.plot(epochs, train_losses, label='train loss')
        ax.plot(epochs, test_losses, label='test loss')
        
        # add batch variation if provided
        if batch_variation:
            train_batch_mins, train_batch_maxs = batch_variation
            ax.fill_between(epochs, train_batch_mins, train_batch_maxs, 
                          color='lightsteelblue', alpha=0.3, label='batch variation')
        
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title('training and test losses')
        ax.legend()
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.4f}'))
        ax.set_xticks(list(epochs)[::max(len(epochs) // 20, 1)])
        ax.grid(True)
        sns.despine()

    @staticmethod
    def plot_metrics(ax, epochs: List[int], metrics_dict: dict):
        """plots training metrics."""
        for metric_name, values in metrics_dict.items():
            ax.plot(epochs, values, label=metric_name)
        
        ax.set_xlabel('epochs')
        ax.set_ylabel('metric')
        ax.set_title('training metrics')
        ax.legend()
        ax.set_xticks(list(epochs)[::max(len(epochs) // 20, 1)])
        ax.grid(True)
        sns.despine()