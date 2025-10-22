from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import warnings

# Suppress the specific UserWarning from pkg_resources that is triggered by multiprocessing.
# This warning is not actionable from within this codebase and clutters the output.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.", category=UserWarning)

score_names = {
    "balanced_accuracy": "Balanced Accuracy",
    "precision_weighted": "Weighted Precision",
    "recall_weighted": "Weighted Recall",
    "f1_none": "F1-Score Per Class",
    "f1_weighted": "Weighted F1-Score"
}

def create_directory(dir_path):
    """Safely creates a directory and its parents.

    Args:
        dir_path: The path to the directory to create.

    Returns:
        The path if the directory was created successfully or already existed,
        otherwise None.
    """
    if not dir_path:
        return None
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return dir_path
    except OSError:
        # Could not create the directory, e.g., due to permissions.
        # Depending on requirements, you might want to log this error.
        return None
    
def merge_consecutive_records(df, identifier_column, sort_column='Parity', target_column='Liveborn'):
    """Merges consecutive records for each unique identifier.
    
    For each group specified by `identifier_column` (e.g., for each sow), this
    function sorts by `sort_column` to find consecutive records. For each pair
    of consecutive records, it takes the first record and creates a new 'Regression'
    column containing the `target_column` value from the second (i.e., next) record.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        identifier_column (str): The column to group by (e.g., 'Sow ID').
        sort_column (str): The column to sort by to determine consecutiveness (e.g., 'Parity').
        target_column (str): The source column from the next record whose value will be
            placed in the new 'Regression' column.

    Returns:
        pd.DataFrame: A new DataFrame containing the merged records.
    """
    # Drop duplicates to ensure each (identifier, sort_column) pair is unique.
    # This prevents issues with multiple records for the same parity and ensures
    # the logic matches the original iterative approach.
    df_unique = df.drop_duplicates(subset=[identifier_column, sort_column])
    # Ensure the DataFrame is sorted to process records in the correct order within each group
    df_sorted = df_unique.sort_values(by=[identifier_column, sort_column]).copy()
    
    # Within each group (sow), create columns for the next record's data.
    grouped = df_sorted.groupby(identifier_column)
    df_sorted['next_sort_val'] = grouped[sort_column].shift(-1)
    df_sorted[f'next_{target_column}'] = grouped[target_column].shift(-1)
    
    # Create a 'Regression' column, initially filled with NaN
    df_sorted[f'{target_column} (Next Parity)'] = np.nan
    
    # Create a mask to identify rows where the next record is truly consecutive
    consecutive_mask = (df_sorted['next_sort_val'] == df_sorted[sort_column] + 1)
    
    # Use the mask to assign the 'next_Liveborn' value to the 'Regression' column only for consecutive records
    df_sorted.loc[consecutive_mask, f'{target_column} (Next Parity)'] = df_sorted.loc[consecutive_mask, f'next_{target_column}'].values
    
    # Drop rows that were not part of a consecutive pair (where 'Regression' is still NaN) and clean up helper columns
    merged_df = df_sorted.dropna(subset=[f'{target_column} (Next Parity)']).drop(columns=['next_sort_val', f'next_{target_column}'])
    
    return merged_df.astype({f'{target_column} (Next Parity)': int}).reset_index(drop=True)

def bin(arr, inner_quantiles, labels):
    """Bins an array-like object into discrete intervals based on quantiles.

    This function uses pandas.qcut, which is designed for quantile-based
    discretization. It automatically handles cases where quantile boundaries
    are not unique by merging intervals.

    Args:
        arr (pd.Series or array-like): The input data to be binned. Must be
            a pandas Series to use the .quantile() method.
        inner_quantiles (list of floats): A list of inner quantile points to create
            the bins (e.g., [0.25, 0.75]).
        labels (list of str): The labels for the returned bins. The number of
            labels must be one more than the number of inner_quantiles.

    Returns:
        pd.Series: A pandas Series containing the labeled bins for each
            element in the input array.
    """
    if len(labels) != len(inner_quantiles) + 1:
        raise ValueError("Number of labels must be one more than the number of inner quantiles.")
    
    # Construct the full list of quantiles for the bin edges
    q = [0] + inner_quantiles + [1]
    return pd.qcut(arr, q=q, labels=labels)

def summary_statistics(df, group_column=None):
    """
    Calculates summary statistics for both numeric and categorical columns of a
    DataFrame.

    For numeric columns, it computes: min, mean, median, max, and standard deviation.
    For categorical columns, it computes the count and percentage of each category.

    Args:
        df (pd.DataFrame): The input DataFrame.
        group_column (str): The name of the column to group by. Defaults to None.

    Returns:
        pd.DataFrame: A multi-index DataFrame containing the summary statistics
                      for each column, for each group.
    """
    data_source = df.groupby(group_column, observed=False) if group_column else df
    results = {}

    # Identify numeric and categorical columns (excluding the group column)
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(group_column, errors='ignore')
    categorical_cols = df.select_dtypes(include=['category']).columns.drop(group_column, errors='ignore')

    # Calculate stats for numeric columns
    if not numeric_cols.empty:
        numeric_stats = data_source[numeric_cols].agg(['min', 'mean', 'median', 'max', 'std'])
        results['numerical'] = numeric_stats

    # Calculate stats for categorical columns
    if not categorical_cols.empty:
        # Process each categorical column individually to handle different value sets
        # and correctly build a multi-indexed DataFrame.
        all_stats = []
        for col in categorical_cols:
            counts = data_source[col].value_counts(normalize=False).sort_index()
            percentages = data_source[col].value_counts(normalize=True).sort_index()
            
            # Combine counts and percentages for the current column
            stats = pd.concat([counts, percentages], axis=1, keys=['count', 'percentage'])
            all_stats.append(stats)

        # Combine stats from all columns and set 'column' as the top-level index
        categorical_stats = pd.concat(all_stats)
        if group_column:
            categorical_stats = categorical_stats.reorder_levels([0, 1])
        results['categorical'] = categorical_stats

    return results

def ensemble(estimators, exclusion_list=[], param_grid=None):
    """Creates an ensemble of estimators by excluding specified ones.

    Args:
        estimators (dict): A dictionary of estimator name to estimator object.
        exclusion_list (list): A list of estimator names to exclude from the ensemble.

    Returns:
        dict: A dictionary of estimator name to estimator object for the ensemble.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from model import Estimator
    
    clf_list = [(name, clf.estimator) for name, clf in estimators.items() if name not in exclusion_list]
    est_params = next(iter(estimators.values())).params  # Get parameters from any estimator

    voting = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', VotingClassifier(estimators=clf_list, n_jobs=est_params.n_jobs))
    ])
    stacking = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', StackingClassifier(estimators=clf_list, final_estimator=LogisticRegression(max_iter=100000, random_state=42), n_jobs=est_params.n_jobs))
    ])
    
    estimators['Voting'] = Estimator(voting, 'Voting', est_params, param_grid)
    estimators['Stacking'] = Estimator(stacking, 'Stacking', est_params, param_grid)
    
    return estimators

def process_evaluation_scores(clf_name, default_scores=None, optimized_scores=None, title="Evaluation Scores", verbose=True):
    """
    Prints and structures a comparison of default and optimized evaluation scores.

    Args:
        clf_name (str): The name of the classifier.
        default_scores (dict): The dictionary of scores for the default model.
        optimized_scores (dict): The dictionary of scores for the optimized model.
        title (str): A title for the printed output section.
        verbose (bool, optional): If True, prints the scores to the console.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the formatted scores.
    """
    if default_scores is None and optimized_scores is None:
        if verbose:
            print(f"\n--- {clf_name}: {title} ---")
            print("No scores provided to process.")
        return pd.DataFrame()

    show_default = default_scores is not None
    show_optimized = optimized_scores is not None
    show_both = show_default and show_optimized
    
    if verbose: # Prepare and print header
        print(f"\n--- {clf_name}: {title} ---")
        if show_both:
            header = f"{'Metric':<25} {'Default':<22} | {'Optimized':<22}"
        elif show_optimized:
            header = f"{'Metric':<25} {'Optimized Score':<22}"
        else:
            header = f"{'Metric':<25} {'Default Score':<22}"
        print(header)
        print("-" * len(header))
        
    # Determine which scorers to process
    default_keys = set(default_scores.keys()) if default_scores else set()
    optimized_keys = set(optimized_scores.keys()) if optimized_scores else set()
    all_scorer_names = sorted(default_keys | optimized_keys)
    results_list = []

    for scorer_name in all_scorer_names:
        if scorer_name == 'confusion_matrices':
            continue

        # Format metric name for display
        if scorer_name.startswith('cv_score_'):
            pretty_name = f"CV Score ({score_names.get(scorer_name.replace('cv_score_', ''), scorer_name)})"
        else:
            pretty_name = score_names.get(scorer_name, scorer_name)
        if verbose:
            print(f"{pretty_name}:")

        def get_score_stats(scores):
            """Calculates mean and std, returns them and a formatted string."""
            if scores is None or len(scores) == 0:
                return np.nan, np.nan, "N/A" # Return tuple of 3
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            return mean_score, std_score, f"{mean_score:.2f} ± {std_score:.2f}"

        default_metric_scores = default_scores.get(scorer_name) if show_default else None
        optimized_metric_scores = optimized_scores.get(scorer_name) if show_optimized else None

        # Consolidate handling for per-class (dict) and overall (list) scores
        if isinstance(default_metric_scores, dict) or isinstance(optimized_metric_scores, dict):
            # Per-class scores
            d_scores = default_metric_scores or {}
            o_scores = optimized_metric_scores or {}
            all_labels = sorted(set(d_scores.keys()) | set(o_scores.keys()))
            score_items = {label: (d_scores.get(label), o_scores.get(label)) for label in all_labels}
        else:
            # Overall scores
            score_items = {'Overall': (default_metric_scores, optimized_metric_scores)}

        for label, (d_score, o_score) in score_items.items():
            _, _, default_str = get_score_stats(d_score)
            _, _, optimized_str = get_score_stats(o_score)

            if verbose:
                label_str = f"  - {label:<21}" if label != 'Overall' else f"  - {'Overall':<21}"
                if show_both:
                    print(f"{label_str} {default_str:<22} | {optimized_str:<22}")
                elif show_optimized:
                    print(f"{label_str} {optimized_str:<22}")
                elif show_default:
                    print(f"{label_str} {default_str:<22}")

            record = {'Classifier': clf_name, 'Metric': pretty_name, 'Class': label}
            if show_both:
                record.update({'Default_Score': default_str, 'Optimized_Score': optimized_str})
            else:
                record['Score'] = optimized_str if show_optimized else default_str
            results_list.append(record)

    if verbose:
        print()  # Add a newline for better separation
    
    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(results_list)

def test_normality(df, group_column, alpha=0.05, verbose=True):
    """
    Performs the Shapiro-Wilk test for normality on numeric columns, grouped by a categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        group_column (str): The name of the column to use for grouping the data.
        alpha (float, optional): The significance level for the test.
            Defaults to 0.05.
        verbose (bool, optional): If True, prints detailed test results to the
            console. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame where columns are the numeric features, rows are
                      the unique classes from the group_column, and cells contain
                      'Yes' (normal), 'No' (not normal), or 'N/A' (not enough data).
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.drop(group_column, errors='ignore')
    
    # Use categories if the column is categorical to respect order, otherwise sort unique values
    if pd.api.types.is_categorical_dtype(df[group_column]):
        unique_classes = df[group_column].cat.categories
    else:
        unique_classes = sorted(df[group_column].unique())
        
    if verbose:
        print("\n--- Shapiro-Wilk Test Details ---")
    results_for_df = {}

    for col in numeric_cols:
        results_for_df[col] = {}
        if verbose:
            print(f"\nVariable: {col}")
        for cls in unique_classes:
            data_subset = df[df[group_column] == cls][col]

            if len(data_subset) < 3:
                results_for_df[col][cls] = 'N/A'
                if verbose:
                    print(f"  - Class '{cls}': Not enough data to test (n={len(data_subset)})")
                continue

            stat, p_value = shapiro(data_subset)
            is_normal = p_value > alpha
            results_for_df[col][cls] = 'Yes' if is_normal else 'No'
            
            if verbose:
                print(f"  - Class '{cls}': Statistic={stat:.4f}, p-value={p_value:.4f} -> {'Normal' if is_normal else 'Not Normal'}")

    if verbose:
        print("-" * 35) # Separator after the detailed prints
    return pd.DataFrame(results_for_df)

def synthesize_dataset(dataset_type, n_sows=100, max_parities=7, n_farms=5, random_state=None):
    """
    Generates a synthetic dataset mimicking the structure of 'cdpq' or 'hypor' raw data.

    Args:
        dataset_type (str): The type of dataset to synthesize, either 'cdpq' or 'hypor'.
        n_sows (int, optional): The number of unique sows to generate. Defaults to 100.
        max_parities (int, optional): The maximum number of consecutive parities for any sow.
            Defaults to 7.
        n_farms (int, optional): The number of farms to simulate for the 'hypor' dataset.
            Defaults to 5.
        random_state (int, optional): Seed for the random number generator for reproducibility.
            Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the synthetic data.

    Raises:
        ValueError: If an unsupported dataset_type is provided.
    """
    rng = np.random.default_rng(random_state)
    records = []

    for sow_id in range(1, n_sows + 1):
        num_parities = rng.integers(1, max_parities + 1)
        for parity in range(1, num_parities + 1):
            record = {
                "Sow ID": f"Sow_{sow_id}",
                "Parity": parity,
                "Gestation Length": rng.normal(115, 2),
                "Lactation Length": rng.normal(21, 3),
                "Stillborn": rng.poisson(1.2),
                "Mummies": rng.poisson(0.5),
                "Piglets Weaned": rng.integers(9, 14),
                "Liveborn": rng.integers(10, 18)
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Add dataset-specific columns and formatting
    if dataset_type == 'cdpq':
        df["Farm"] = 1
        # Simulate body weight and backfat measurements
        df["Breeding Weight"] = rng.normal(230, 25, size=len(df))
        df["Farrowing Weight"] = df["Breeding Weight"] + rng.normal(30, 10, size=len(df))
        df["Weaning Weight"] = df["Farrowing Weight"] - rng.normal(20, 8, size=len(df))
        df["Breeding Backfat"] = rng.normal(17, 3, size=len(df))
        df["Farrowing Backfat"] = df["Breeding Backfat"] + rng.normal(2, 1, size=len(df))
        df["Weaning Backfat"] = df["Farrowing Backfat"] - rng.normal(1.5, 1, size=len(df))

    elif dataset_type == 'hypor':
        # Simulate multiple farms
        sow_farm_map = {f"Sow_{sow_id}": f"Farm_{rng.integers(1, n_farms + 1)}" for sow_id in range(1, n_sows + 1)}
        df["Farm"] = df["Sow ID"].map(sow_farm_map)

    else:
        raise ValueError(f"Unsupported dataset_type: '{dataset_type}'. Choose 'cdpq' or 'hypor'.")

    # Round numeric columns to reasonable precision
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)
        else:
            df[col] = df[col].astype(int)

    return df
