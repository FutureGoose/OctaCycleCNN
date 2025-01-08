import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class MetricsPlotter:
    """handles all plotting functionality for training metrics."""
    
    @staticmethod
    def plot_losses(ax, epochs: List[int], train_losses: List[float], 
                   val_losses: List[float], batch_variation: Optional[tuple] = None):
        """Plots training and validation losses."""
        # plot main loss lines
        ax.plot(epochs, train_losses, label='train loss')
        ax.plot(epochs, val_losses, label='val loss')
        
        # add batch variation if provided
        if batch_variation:
            train_batch_mins, train_batch_maxs = batch_variation
            ax.fill_between(epochs, train_batch_mins, train_batch_maxs, 
                          color='lightsteelblue', alpha=0.3, label='batch variation')
        
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title('training and validation losses')
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

    @staticmethod
    def plot_class_accuracy(accuracies: np.ndarray, class_names: List[str], 
                          dataset_name: str = "Dataset", ax: Optional[plt.Axes] = None, 
                          figsize: tuple = (10, 6)) -> plt.Axes:
        """Plots per-class accuracy as horizontal bars.
        
        Args:
            accuracies (numpy.ndarray): array of accuracy values per class
            class_names (list): list of class names
            dataset_name (str): name of dataset for title
            ax (matplotlib.axes, optional): axes to plot on
            figsize (tuple): figure size if creating new plot
            
        Returns:
            matplotlib.axes: the axes object with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        # create data frame for plotting
        df = pd.DataFrame({
            'Class': class_names,
            'Accuracy': accuracies
        })
        
        # use a more subtle color palette
        colors = sns.color_palette("muted", n_colors=len(class_names))
        
        # create horizontal bar plot
        sns.barplot(
            data=df,
            y='Class',
            x='Accuracy',
            ax=ax,
            orient='h',
            hue='Class',
            palette=colors,
            legend=False
        )
        
        # customize appearance
        ax.set_title(f'Per-Class Accuracy on {dataset_name} Test Set')
        ax.set_xlabel('Accuracy (%)')
        ax.set_xlim(0, 100)
        
        # add percentage labels on bars
        for i, v in enumerate(accuracies):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        
        sns.despine()
        plt.tight_layout()
        
        return ax

    @staticmethod
    def plot_confusion_matrix(predictions: np.ndarray, true_labels: np.ndarray, 
                            class_names: List[str], dataset_name: str = "Dataset",
                            ax: Optional[plt.Axes] = None, 
                            figsize: tuple = (10, 8),
                            normalize: bool = True) -> plt.Axes:
        """Plots confusion matrix as a heatmap.
        
        Args:
            predictions (numpy.ndarray): model predictions
            true_labels (numpy.ndarray): ground truth labels
            class_names (list): list of class names
            dataset_name (str): name of dataset for title
            ax (matplotlib.axes, optional): axes to plot on
            figsize (tuple): figure size if creating new plot
            normalize (bool): whether to normalize confusion matrix
            
        Returns:
            matplotlib.axes: the axes object with the plot
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
            
        # compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.1%'
        else:
            fmt = 'd'
            
        # create heatmap
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='YlOrRd',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax)
        
        # customize appearance
        ax.set_title(f'Confusion Matrix - {dataset_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        return ax