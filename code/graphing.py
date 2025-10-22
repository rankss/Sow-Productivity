import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils import create_directory
import pandas as pd

def plot_learning_curve(train_sizes, train_scores, test_scores, title, xlabel, ylabel, figure_directory=None):
    """Plots a learning curve."""
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.plot(train_sizes, train_scores_mean, label="Training score", color='red')
    ax.plot(train_sizes, test_scores_mean, label="Cross-validation score", color='green')

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='red')
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='green')

    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([np.min(train_sizes), np.max(train_sizes)])
    ax.grid()
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')
    
    fig.tight_layout()
    
    if figure_directory:
        save_dir = os.path.join(figure_directory, 'learning_curves')
        create_directory(save_dir)
        fig.savefig(os.path.join(save_dir, f'lc_{title.replace(" ", "_")}.png'), dpi=300)
        print(f"Learning curve saved to: {os.path.join(save_dir, f'lc_{title.replace(' ', '_')}.png')}")
        
    plt.close(fig)

def plot_distributions(df, group_column, figure_directory=None):
    """
    Generates and saves Kernel Density Estimate (KDE) plots for numeric columns,
    grouped by a specified categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        group_column (str): The name of the column to use for grouping and
            coloring the plots.
        figure_directory (str, optional): The base directory where plots will be
            saved. Plots are stored in a 'distribution_plots' subdirectory. If None,
            plots are not saved. Defaults to None.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(group_column, errors='ignore')
    
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.kdeplot(data=df, x=col, hue=group_column, common_norm=False, alpha=0.2, ax=ax, multiple='layer', fill=True, legend=False)
        ax.set_xlabel('')
        ax.set_ylabel('Density')
        fig.tight_layout()
        
        if figure_directory:
            save_dir = os.path.join(figure_directory, 'distribution_plots')
            create_directory(save_dir)
            save_path = os.path.join(save_dir, f'kde_{col}.png')
            fig.savefig(save_path, dpi=300)
            print(f"Distribution plot saved to: {save_path}")
        plt.close(fig)
        
def plot_confusion_matrices(confusion_matrices, class_labels, normalize='true', figure_directory=None):
    """
    Plots confusion matrices for multiple classifiers in a single figure.

    This function takes a dictionary containing the confusion matrices for a
    set of classifiers. It calculates the mean and standard deviation for each
    and plots them as heatmaps in a grid.

    Args:
        confusion_matrices (dict): A dictionary where keys are the model states
            (e.g., 'DT', 'RF') and values are lists of confusion matrix arrays
            from repeated evaluations.
        class_labels (dict): A dictionary mapping integer class codes to string
            labels (e.g., {0: 'Low', 1: 'Medium'}).
        normalize ({'true', 'pred', None}, optional): The normalization method.
            - 'true': Normalizes over the true labels (rows).
            - 'pred': Normalizes over the predicted labels (columns).
            - None: No normalization (raw counts).
            Defaults to 'true'.
        figure_directory (str, optional): Directory to save the plot. If None,
            the plot is not saved. Defaults to None.
    """
    # Derive ordered class names from the dictionary for labeling
    class_names = [class_labels[k] for k in sorted(class_labels.keys())]
    
    if not confusion_matrices:
        print("Warning: confusion_matrices dictionary is empty. Skipping plot.")
        return

    # Arrange plots in a grid with 3 columns
    n_classifiers = len(confusion_matrices)
    n_cols = 3
    n_rows = int(np.ceil(n_classifiers / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax, (clf_name, matrices) in zip(axes, confusion_matrices.items()):
        matrices = confusion_matrices.get(clf_name, [])
        if not matrices:
            title_prefix = 'Normalized ' if normalize else ''
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Mean {title_prefix}CM: {clf_name}')
            continue

        matrices_arr = np.array(matrices)

        if normalize:
            with np.errstate(divide='ignore', invalid='ignore'):
                if normalize == 'true':
                    # Sum over columns for each matrix (axis=2)
                    cm_sum = matrices_arr.sum(axis=2, keepdims=True)
                elif normalize == 'pred':
                    # Sum over rows for each matrix (axis=1)
                    cm_sum = matrices_arr.sum(axis=1, keepdims=True)
                else:
                    raise ValueError("normalize must be one of {'true', 'pred', None}")
                
                processed_matrices = np.divide(matrices_arr, cm_sum, out=np.zeros_like(matrices_arr, dtype=float), where=cm_sum!=0)

            mean_cm = np.mean(processed_matrices, axis=0)
            std_cm = np.std(processed_matrices, axis=0)
            title_prefix, file_prefix = 'Normalized ', f'normalized_{normalize}_'
        else:
            mean_cm = np.mean(matrices_arr, axis=0)
            std_cm = np.std(matrices_arr, axis=0)
            title_prefix, file_prefix = '', ''

        # Set title and filename components
        plot_title = f'Mean {title_prefix}CM: {clf_name}'

        sns.heatmap(mean_cm, annot=False, fmt='.2f', cmap='Blues', ax=ax, cbar=True)

        # Add custom annotations for mean and std dev
        for (i, j), mean_val in np.ndenumerate(mean_cm):
                std_val = std_cm[i, j]
                text_color = 'white' if mean_val > np.max(mean_cm) / 2 else 'black'
                ax.text(j + 0.5, i + 0.35, f'{mean_val:.2f}',
                        ha='center', va='center', color=text_color, fontsize=12)
                ax.text(j + 0.5, i + 0.65, f'±{std_val:.2f}',
                        ha='center', va='center', color=text_color, fontsize=9)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(plot_title)
        ax.set_xticklabels(class_names, rotation=0)
        ax.set_yticklabels(class_names, rotation=0)

    # Hide any unused subplots
    for i in range(n_classifiers, len(axes)):
        axes[i].set_visible(False)

    title_prefix = 'Normalized ' if normalize else ''
    fig.suptitle(f'Mean {title_prefix}Confusion Matrices for Optimized Models', fontsize=16, y=1.02)
    fig.tight_layout()

    if figure_directory:
        save_dir = os.path.join(figure_directory, 'confusion_matrices')
        create_directory(save_dir)
        
        file_prefix = f'normalized_{normalize}_' if normalize else ''
        filename = f'cm_summary_{file_prefix}optimized.png'
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300)
        print(f"Confusion matrix plot saved to: {save_path}")
    plt.close(fig)
    
def plot_feature_importance(all_importances_dict, feature_names, title, figure_directory=None):
    """
    Plots permutation feature importances for multiple classifiers in a single figure.

    Args:
        all_importances_dict (dict): A dictionary where keys are classifier names
            and values are the result objects from sklearn.inspection.permutation_importance.
        feature_names (list): The names of the features corresponding to the
            importance scores.
        title (str): The title for the plot.
        figure_directory (str, optional): Directory to save the plot. If None,
            the plot is not saved. Defaults to None.
    """
    if not all_importances_dict:
        print("Warning: Feature importance dictionary is empty. Skipping plot.")
        return

    # Arrange plots in a grid with 3 columns
    n_classifiers = len(all_importances_dict)
    n_cols = 3
    n_rows = int(np.ceil(n_classifiers / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows), sharex=True)
    axes = axes.flatten() # Flatten the 2D array of axes for easy iteration


    # Determine the global x-axis range for consistent comparison
    min_importance, max_importance = 0, 0
    for result in all_importances_dict.values():
        min_importance = min(min_importance, np.min(result.importances_mean - result.importances_std))
        max_importance = max(max_importance, np.max(result.importances_mean + result.importances_std))

    for ax, (clf_name, importance_result) in zip(axes, all_importances_dict.items()):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': importance_result.importances_mean,
            'importance_std': importance_result.importances_std
        }).sort_values('importance_mean', ascending=True)

        ax.barh(
            importance_df['feature'],
            importance_df['importance_mean'],
            xerr=importance_df['importance_std'],
            align='center',
            capsize=4
        )
        ax.set_title(f"Feature Importance for {clf_name}")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Hide any unused subplots
    for i in range(n_classifiers, len(axes)):
        axes[i].set_visible(False)

    # Apply global settings
    fig.suptitle(title, fontsize=16, y=1.02)
    # Set shared x-axis properties on the last row of plots
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Permutation Importance (Decrease in weighted F1-Score)")
    fig.tight_layout()

    if figure_directory:
        save_dir = os.path.join(figure_directory, 'feature_importance')
        create_directory(save_dir)
        filename = f'fi_summary_{title.replace(" ", "_")}.png'
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300)
        print(f"Feature importance plot saved to: {save_path}")

    plt.close(fig)

def plot_all_classifier_scores(scores_df, metric_name, title, figure_directory=None):
    """
    Creates a bar plot to compare a single metric across all classifiers.

    Args:
        scores_df (pd.DataFrame): A DataFrame containing the evaluation scores
            for all classifiers. Expected columns include 'Classifier', 'Metric',
            'Optimized_Mean', and 'Optimized_Std'.
        metric_name (str): The name of the metric to plot (e.g., 'F1-Score (Weighted)').
        title (str): The title for the plot.
        figure_directory (str, optional): Directory to save the plot. If None,
            the plot is not saved. Defaults to None.
    """
    # Filter for the specific metric and 'Overall' class scores
    metric_df = scores_df[(scores_df['Metric'] == metric_name) & (scores_df['Class'] == 'Overall')]
    metric_df = metric_df.sort_values('Optimized_Mean', ascending=False)
    
    print(metric_df)

    if metric_df.empty:
        print(f"Warning: No data found for metric '{metric_name}'. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Pass the data Series directly to x and y to prevent seaborn's internal aggregation.
    # This ensures that the shape of y matches the shape of yerr.
    sns.barplot(
        x=metric_df['Classifier'], 
        y=metric_df['Optimized_Mean'], 
        ax=ax, capsize=.1, legend=False)
    
    ax.set_ylim([0, 1])
    ax.set_title(title)
    ax.set_ylabel(f"Mean Score ({metric_name})")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='x')
    fig.tight_layout()

    if figure_directory:
        save_dir = os.path.join(figure_directory, 'summary_plots')
        create_directory(save_dir)
        filename = f'summary_{metric_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=300)
        print(f"Summary score plot saved to: {save_path}")

    plt.close(fig)