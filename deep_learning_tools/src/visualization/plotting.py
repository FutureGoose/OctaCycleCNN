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
                            normalize: bool = True,
                            cmap: str = 'Blues') -> plt.Axes:
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
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
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

    @staticmethod
    def plot_classification_examples(predictions: np.ndarray, 
                                   true_labels: np.ndarray,
                                   probabilities: np.ndarray,
                                   images: np.ndarray,
                                   class_names: List[str],
                                   scenarios: List[dict],
                                   n_cols: int = 3,
                                   figsize: tuple = None,
                                   mean: tuple = (0.4914, 0.4822, 0.4465),
                                   std: tuple = (0.2470, 0.2435, 0.2616)) -> plt.Figure:
        """Plots examples of model predictions based on specified scenarios.
        
        Args:
            predictions: array of model predictions
            true_labels: array of true labels
            probabilities: array of prediction probabilities
            images: array of images
            class_names: list of class names
            scenarios: list of dictionaries, each containing:
                - true: true class name or '*' for any
                - pred: predicted class name or '*' for any except true
                - n: number of examples to show
            n_cols: number of columns in the plot grid
            figsize: figure size, auto-calculated if None
            mean: mean values used for normalization (CIFAR-10 default)
            std: std values used for normalization (CIFAR-10 default)
            
        Returns:
            matplotlib.figure.Figure: the figure object
        """
        # convert class names to indices
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        # helper function to get matching indices for a scenario
        def get_scenario_indices(scenario: dict) -> np.ndarray:
            true_class = scenario['true']
            pred_class = scenario['pred']
            
            # get indices based on true class
            if true_class == '*':
                true_mask = np.ones_like(true_labels, dtype=bool)
            else:
                true_mask = true_labels == class_to_idx[true_class]
            
            # get indices based on predicted class
            if pred_class == '*':
                if true_class != '*':
                    # exclude correct predictions when true class is specified
                    pred_mask = predictions != class_to_idx[true_class]
                else:
                    pred_mask = np.ones_like(predictions, dtype=bool)
            else:
                pred_mask = predictions == class_to_idx[pred_class]
            
            # combine masks
            return np.where(true_mask & pred_mask)[0]
        
        # collect examples for each scenario
        scenario_examples = []
        for scenario in scenarios:
            indices = get_scenario_indices(scenario)
            if len(indices) == 0:
                print(f"Warning: No examples found for scenario {scenario}")
                continue
                
            # randomly select n examples
            n_examples = min(scenario['n'], len(indices))
            selected_indices = np.random.choice(indices, size=n_examples, replace=False)
            scenario_examples.append({
                'indices': selected_indices,
                'scenario': scenario
            })
        
        # calculate grid layout
        total_examples = sum(len(ex['indices']) for ex in scenario_examples)
        if total_examples == 0:
            raise ValueError("No examples found for any scenario")
            
        n_cols = min(n_cols, total_examples)
        n_rows = (total_examples + n_cols - 1) // n_cols
        
        # calculate figure size if not provided, with extra space for titles
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)  # increased height multiplier from 3 to 4
        
        # create figure with extra space between subplots
        fig = plt.figure(figsize=figsize)
        plt.subplots_adjust(hspace=0.4)  # increase vertical space between subplots
        
        # plot each example
        plot_idx = 1
        for scenario_data in scenario_examples:
            scenario = scenario_data['scenario']
            indices = scenario_data['indices']
            
            for idx in indices:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                
                # get and denormalize image
                img = images[idx]
                img = MetricsPlotter.denormalize_image(img, mean, std)  # denormalize before transpose
                if img.shape[0] == 3:  # if channels first, transpose
                    img = img.transpose(1, 2, 0)
                
                # plot image
                ax.imshow(img)
                
                # get prediction info
                true_class = class_names[true_labels[idx]]
                pred_class = class_names[predictions[idx]]
                prob = probabilities[idx][predictions[idx]]
                
                # set title with smaller font and more padding
                title = f'True: {true_class}\nPred: {pred_class}\nConf: {prob:.2f}'
                ax.set_title(title, fontsize=9, pad=5)
                ax.axis('off')
                
                plot_idx += 1
        
        plt.tight_layout()
        return fig

    @staticmethod
    def denormalize_image(img: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
        """Denormalizes an image using given mean and std values.
        
        Args:
            img: normalized image array
            mean: channel-wise mean values used for normalization
            std: channel-wise std values used for normalization
            
        Returns:
            denormalized image array
        """
        img = img.copy()
        for i in range(len(mean)):
            img[i] = img[i] * std[i] + mean[i]
        return np.clip(img, 0, 1)