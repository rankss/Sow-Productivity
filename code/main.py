# -------------------------------------------------------------------
# Step 1. Load the datasets
# -------------------------------------------------------------------
import pandas as pd
import numpy as np
from utils import synthesize_dataset

use_synthetic_data = True
dataset_name = "cdpq" # "hypor"
subset = False # For CDPQ Dataset Only

if use_synthetic_data:
    df = synthesize_dataset(dataset_type=dataset_name, n_sows=500, max_parities=8, n_farms=10, random_state=42)
    print(f"--- Using synthetic '{dataset_name}' dataset ---")
else:
    dataset_path = f"raw_data/{dataset_name}_raw_dataset.xlsx"
    try:
        df = pd.read_excel(dataset_path)
    except FileNotFoundError:
        print(f"Data file not found at: {dataset_path}")
    print(f"--- Using real '{dataset_name}' dataset ---")

figure_directory = f"figures/{dataset_name}/"
output_directory = f"outputs/{dataset_name}/"

# -------------------------------------------------------------------
# Step 2. Filtering and preprocessing of the CDPQ and Hypor Datasets
# -------------------------------------------------------------------
from utils import merge_consecutive_records, bin

# --- Preprocessing CDPQ Dataset ---
# Rename columns
if dataset_name == "cdpq":
    df = df.rename(columns={
        "Breeding Weight": "Breeding BW",
        "Farrowing Weight": "Farrowing BW",
        "Weaning Weight": "Weaning BW",
        "Breeding Backfat": "Breeding BFT",
        "Farrowing Backfat": "Farrowing BFT",
        "Weaning Backfat": "Weaning BFT"
    })

    # Setting farm and rearranging columns
    df['Farm'] = 1
    df = df[["Sow ID", "Farm", "Parity", "Gestation Length", "Lactation Length", 
             "Stillborn", "Mummies", "Piglets Weaned", "Breeding BW", "Farrowing BW", 
             "Weaning BW", "Breeding BFT", "Farrowing BFT", "Weaning BFT", "Liveborn"]]
    
    if subset:
        df = df.drop(columns=["Breeding BW", "Farrowing BW", 
             "Weaning BW", "Breeding BFT", "Farrowing BFT", "Weaning BFT"])
        
        dataset_name = "cdpq_subset"
        figure_directory = f"figures/{dataset_name}"
        output_directory = f"outputs/{dataset_name}"
        
        df.to_csv(f"processed_data/{dataset_name}_processed_dataset.csv", index=False)
        print(f"Processed dataset saved at: processed_data/{dataset_name}_processed_dataset.csv")
    
else:
    df['Farm'] = df['Farm'].astype('category').copy()
    df['Farm'] = df['Farm'].cat.codes + 1
    df = df[["Sow ID", "Farm", "Parity", "Gestation Length", "Lactation Length",
             "Stillborn", "Mummies", "Piglets Weaned", "Liveborn"]]
    
df['Sow ID'] = df['Sow ID'].astype('object').copy()
df['Farm'] = df['Farm'].astype('category').copy()
df = merge_consecutive_records(df, 'Sow ID')

class_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
class_labels_int = list(class_labels.keys())
class_labels_str = list(class_labels.values())

df['Classification'] = bin(df['Liveborn (Next Parity)'], inner_quantiles=[0.25, 0.75], labels=class_labels_int)
df["Classification"] = df["Classification"].astype('category').copy()
df = df.dropna()

X = df[df.columns[2:-2]]
y = df['Classification'] # Use integer codes for the target variable
group_col = 'Farm' if dataset_name == "hypor" else 'Sow ID'
group = df[group_col]

# --- Save the processed dataset ---
processed_dataset_path = f"processed_data/{dataset_name}_processed_dataset.csv"
df.to_csv(processed_dataset_path, index=False)
print(f"Processed dataset saved at: {processed_dataset_path}")

# -------------------------------------------------------------------
# Step 2.5 Display Statistical Summaries
# -------------------------------------------------------------------
from utils import summary_statistics, test_normality, create_directory
from graphing import plot_distributions

# --- Statistical Summaries
summary = summary_statistics(df, 'Farm')

# --- Kernel Density Estimation Plots for each variable ---
if not subset:
    plot_distributions(df, group_column='Classification', figure_directory=figure_directory)

# --- Shapiro-Wilk Test of Normality for each variable ---
normality_test_results = test_normality(df, group_column='Classification', verbose=False)

# --- Save normality_test_results to a CSV file ---
create_directory(output_directory)
normality_results_path = f"{output_directory}/normality_test_results.csv"
normality_test_results.to_csv(normality_results_path)
print(f"Normality test results saved to: {normality_results_path}")

# -------------------------------------------------------------------
# Step 3. Machine Learning Algorithm Definitions
# -------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from model import Estimator, EstimatorGlobalParameters

seed = 42
max_iter = int( 1E5 )
# Define the number of parallel jobs. Use a specific number instead of -1
# to prevent memory exhaustion on systems with many cores.
# Adjust this value based on your system's available RAM.
n_jobs = 1
model_params = EstimatorGlobalParameters(
    X, y, group, split_method='random' if 'cdpq' in dataset_name else 'logo', 
    random_state=seed, directory=figure_directory, n_jobs=n_jobs
)

CLASSIFIERS = {
    "DT": DecisionTreeClassifier(random_state=seed),
    "KNN": KNeighborsClassifier(),
    "LR": LogisticRegression(random_state=seed, max_iter=max_iter, n_jobs=n_jobs),
    "MLP": MLPClassifier(random_state=seed, max_iter=max_iter),
    "RF": RandomForestClassifier(random_state=seed, n_jobs=n_jobs),
    "SGD": SGDClassifier(random_state=seed, max_iter=max_iter, n_jobs=n_jobs),
    "SVM": SVC(random_state=seed, max_iter=-1),
}

CLASSIFIER_PARAMS = {'cdpq': {
    "DT":       {'classifier__class_weight': None, 'classifier__criterion': 'gini', 'classifier__max_depth': 1, 'classifier__splitter': 'random'},
    "KNN":      {'classifier__algorithm': 'auto', 'classifier__n_neighbors': 7, 'classifier__p': 1, 'classifier__weights': 'uniform'},
    "LR":       {'classifier__C': np.float64(0.21), 'classifier__class_weight': 'balanced', 'classifier__fit_intercept': True},
    "MLP":      {'classifier__activation': 'relu', 'classifier__alpha': 0.0001, 'classifier__hidden_layer_sizes': (100, 50), 'classifier__learning_rate': 'constant', 'classifier__solver': 'sgd'},
    "RF":       {'classifier__class_weight': 'balanced_subsample', 'classifier__criterion': 'gini', 'classifier__max_depth': 4, 'classifier__n_estimators': 100},
    "SGD":      {'classifier__alpha': 0.01, 'classifier__class_weight': 'balanced', 'classifier__fit_intercept': True, 'classifier__loss': 'log_loss', 'classifier__penalty': 'l1'},
    "SVM":      {'classifier__C': np.float64(1.41), 'classifier__class_weight': 'balanced', 'classifier__degree': 3, 'classifier__gamma': 'scale', 'classifier__kernel': 'poly'}
    }, 'hypor': {
    "DT":       {'classifier__class_weight': 'balanced', 'classifier__criterion': 'gini', 'classifier__max_depth': 3, 'classifier__splitter': 'random'},
    "KNN":      {'classifier__algorithm': 'ball_tree', 'classifier__n_neighbors': 13, 'classifier__p': 1, 'classifier__weights': 'uniform'},
    "LR":       {'classifier__C': np.float64(0.81), 'classifier__class_weight': None, 'classifier__fit_intercept': True},
    "MLP":      {'classifier__activation': 'relu', 'classifier__alpha': 0.001, 'classifier__hidden_layer_sizes': (50, 50), 'classifier__learning_rate': 'constant', 'classifier__solver': 'adam'},
    "RF":       {'classifier__class_weight': None, 'classifier__criterion': 'gini', 'classifier__max_depth': None, 'classifier__n_estimators': 100},
    "SGD":      {'classifier__alpha': 0.01, 'classifier__class_weight': 'balanced', 'classifier__fit_intercept': True, 'classifier__loss': 'modified_huber', 'classifier__penalty': 'l1'},
    "SVM":      {'classifier__C': np.float64(1.41), 'classifier__class_weight': None, 'classifier__degree': 1, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf'}
    }, 'cdpq_subset': {
    "DT":       {'classifier__class_weight': 'balanced', 'classifier__criterion': 'gini', 'classifier__max_depth': None, 'classifier__splitter': 'random'},
    "KNN":      {'classifier__algorithm': 'brute', 'classifier__n_neighbors': 19, 'classifier__p': 2, 'classifier__weights': 'uniform'},
    "LR":       {'classifier__C': np.float64(0.61), 'classifier__class_weight': 'balanced', 'classifier__fit_intercept': True},
    "MLP":      {'classifier__activation': 'relu', 'classifier__hidden_layer_sizes': (50,), 'classifier__solver': 'adam'},
    "RF":       {'classifier__class_weight': None, 'classifier__criterion': 'gini', 'classifier__max_depth': None, 'classifier__n_estimators': 50},
    "SGD":      {'classifier__alpha': 0.01, 'classifier__class_weight': 'balanced', 'classifier__fit_intercept': True, 'classifier__loss': 'log_loss', 'classifier__penalty': 'l1'},
    "SVM":      {'classifier__C': np.float64(1.21), 'classifier__class_weight': 'balanced', 'classifier__degree': 3, 'classifier__gamma': 'scale', 'classifier__kernel': 'poly'}
    }
}

CLASSIFIER_GRIDS = {
    "DT":       {"classifier__criterion": ['gini', 'entropy', 'log_loss'], 
                 "classifier__splitter": ['best', 'random'], 
                 "classifier__max_depth": list(range(1, 5)) + [None],
                 "classifier__class_weight": [None, 'balanced']},

    "KNN":      {"classifier__n_neighbors": list(range(1, 20, 2)),
                 "classifier__algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                 "classifier__weights": ['uniform', 'distance'],
                 "classifier__p": [1, 2]},

    "LR":       {"classifier__fit_intercept": [True, False],
                 "classifier__class_weight": [None, 'balanced'],
                 "classifier__C": list(np.arange(0.01, 1.6, 0.2))},

    "MLP":      {"classifier__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                 "classifier__activation": ['identity', 'logistic', 'tanh', 'relu'],
                 "classifier__solver": ['sgd', 'adam'],},

    "RF":       {"classifier__criterion": ['gini', 'entropy', 'log_loss'],
                 "classifier__max_depth": list(range(1, 5)) + [None],
                 "classifier__n_estimators": [50, 100, 200],
                 "classifier__class_weight": [None, 'balanced', 'balanced_subsample']},

    "SGD":      {"classifier__loss": ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber'],
                 "classifier__fit_intercept": [True, False],
                 "classifier__class_weight": [None, 'balanced'],
                 "classifier__alpha": [0.0001, 0.001, 0.01],
                 "classifier__penalty": ['l2', 'l1', 'elasticnet']},

    "SVM":      {"classifier__C": list(np.arange(0.01, 1.6, 0.2)),
                 "classifier__kernel": ['poly', 'rbf', 'sigmoid', 'linear'],
                 "classifier__degree": list(range(1, 5)),
                 "classifier__gamma": ['scale', 'auto'],
                 "classifier__class_weight": [None, 'balanced']},
}

# -------------------------------------------------------------------
# Step 4. Machine Learning Algorithms Pipeline
# -------------------------------------------------------------------
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.base import clone
from functools import partial
from utils import process_evaluation_scores
from graphing import plot_confusion_matrices, plot_feature_importance, plot_all_classifier_scores

# Suppress the specific UserWarning from pkg_resources that is triggered by multiprocessing.
# This warning is not actionable from within this codebase and clutters the output.
import warnings
warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.", category=UserWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.", category=UserWarning)

guiding_scorer = 'f1_weighted'
scoring_functions = {
    'balanced_accuracy': balanced_accuracy_score,
    'precision_weighted': partial(precision_score, average='weighted', zero_division=0, labels=class_labels_int),
    'recall_weighted': partial(recall_score, average='weighted', zero_division=0, labels=class_labels_int),
    'f1_none': partial(f1_score, average=None, zero_division=0, labels=class_labels_int),
    'f1_weighted': partial(f1_score, average='weighted', zero_division=0, labels=class_labels_int)
}

print("--- Classifier Performance Evaluation ---")

OPT_CLASSIFIERS = {}
all_cv_scores_dfs = []
for clf_name, clf in CLASSIFIERS.items():
    default_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    CLASSIFIERS[clf_name] = Estimator(default_pipe, clf_name, model_params, grid=CLASSIFIER_GRIDS[clf_name])
    CLASSIFIERS[clf_name].fit()
    CLASSIFIERS[clf_name].learning_curves(scoring=guiding_scorer)
    
    # --- Hyperparameter Optimization or Loading Pre-tuned ---
    if dataset_name in CLASSIFIER_PARAMS.keys():
        optimized_pipe = clone(default_pipe)
        optimized_pipe.set_params(**CLASSIFIER_PARAMS[dataset_name][clf_name])
        OPT_CLASSIFIERS[clf_name] = Estimator(optimized_pipe, clf_name, model_params)
        OPT_CLASSIFIERS[clf_name].fit()
        OPT_CLASSIFIERS[clf_name].learning_curves(scoring=guiding_scorer)
    else:
        OPT_CLASSIFIERS[clf_name] = CLASSIFIERS[clf_name].hyperparameter_optimization(scoring=guiding_scorer, search_method='grid')
        OPT_CLASSIFIERS[clf_name].learning_curves(scoring=guiding_scorer)
        
    # --- Cross-Validation and Feature Importance for Default Model ---
    # Create dictionaries to hold CV scores for processing
    default_cv_results = {
        f"cv_score_{guiding_scorer}": CLASSIFIERS[clf_name].cross_validation(scoring=guiding_scorer)
    }
    optimized_cv_results = {
        f"cv_score_{guiding_scorer}": OPT_CLASSIFIERS[clf_name].cross_validation(scoring=guiding_scorer)
    }

    # Use the existing processing function to display CV scores
    cv_scores_df = process_evaluation_scores(clf_name, default_cv_results, optimized_cv_results, title="Cross-Validation Scores")
    all_cv_scores_dfs.append(cv_scores_df)

# -------------------------------------------------------------------
# Step 5. Ensemble Machine Learning Algorithms Pipeline
# -------------------------------------------------------------------
from utils import ensemble

# Define overfitting classifiers for default and optimized classifiers
if dataset_name == "cdpq":
    default_overfit = ['DT', 'KNN', 'MLP', 'RF', 'SVM']
    optimized_overfit = ['KNN', 'MLP', 'RF', 'SVM']
    
if dataset_name == "hypor":
    default_overfit = ['DT', 'KNN', 'RF']
    optimized_overfit = ['KNN', 'RF']
    
if dataset_name == "cdpq_subset":
    default_overfit = ['DT', 'KNN', 'MLP', 'RF', 'SVM']
    optimized_overfit = ['DT', 'MLP', 'RF', 'SVM']

# Create voting and stacking classifiers
CLASSIFIERS = ensemble(CLASSIFIERS, exclusion_list=default_overfit, param_grid=True)
OPT_CLASSIFIERS = ensemble(OPT_CLASSIFIERS, exclusion_list=optimized_overfit)

default_scores = {}
optimized_scores = {}
for clf_name in ['Voting', 'Stacking']:
    CLASSIFIERS[clf_name].fit()
    CLASSIFIERS[clf_name].learning_curves(scoring=guiding_scorer)
    OPT_CLASSIFIERS[clf_name].fit()
    OPT_CLASSIFIERS[clf_name].learning_curves(scoring=guiding_scorer)
    
    # --- Cross-Validation and Feature Importance for Default Model ---
    # Create dictionaries to hold CV scores for processing
    default_cv_results = {
        f"cv_score_{guiding_scorer}": CLASSIFIERS[clf_name].cross_validation(scoring=guiding_scorer)
    }
    optimized_cv_results = {
        f"cv_score_{guiding_scorer}": OPT_CLASSIFIERS[clf_name].cross_validation(scoring=guiding_scorer)
    }

    # Use the existing processing function to display CV scores
    cv_scores_df = process_evaluation_scores(clf_name, default_cv_results, optimized_cv_results, title="Cross-Validation Scores")
    all_cv_scores_dfs.append(cv_scores_df)

# -------------------------------------------------------------------
# Step 6. Final Model Evaluation and Feature Importance
# -------------------------------------------------------------------
all_scores_dfs = []
all_feature_importances = {}
all_optimized_cms = {}
for clf_name in CLASSIFIERS.keys():
    optimized_scores = OPT_CLASSIFIERS[clf_name].repeated_evaluation(scoring_functions=scoring_functions, repeats=100, random_state=seed, class_labels=class_labels)

    # if clf_name not in optimized_overfit:
    all_feature_importances[clf_name] = OPT_CLASSIFIERS[clf_name].permutation_feature_importance(scoring=guiding_scorer, random_state=seed)
    
    # Collect the confusion matrices for the optimized model
    all_optimized_cms[clf_name] = optimized_scores.get('confusion_matrices', [])
    
    # --- Model Evaluation ---
    eval_scores_df = process_evaluation_scores(clf_name, default_scores=None, optimized_scores=optimized_scores, title="Repeated Hold-Out Evaluation")
    all_scores_dfs.append(eval_scores_df)
    
# --- Plot all optimized confusion matrices ---
print("\n--- Generating Combined Confusion Matrix Plot ---")
plot_confusion_matrices(
    all_optimized_cms,
    class_labels=class_labels,
    normalize='true', # Can be 'true', 'pred', or None
    figure_directory=figure_directory
)

# --- Plot all feature importances ---
print("\n--- Generating Combined Feature Importance Plot ---")
plot_feature_importance(
    all_importances_dict=all_feature_importances,
    feature_names=X.columns,
    title="Model Feature Importances",
    figure_directory=figure_directory
)

# --- Save all evaluation scores to a single CSV file ---
final_scores_df = pd.concat(all_scores_dfs, ignore_index=True)
scores_output_path = f"{output_directory}/model_evaluation_scores.csv"
final_scores_df.to_csv(scores_output_path, index=False, encoding='utf-8')
print(f"\nAll model evaluation scores saved to: {scores_output_path}")

# --- Save all cross-validation scores to a single CSV file ---
final_cv_scores_df = pd.concat(all_cv_scores_dfs, ignore_index=True)
scores_output_path = f"{output_directory}/model_cv_scores.csv"
final_cv_scores_df.to_csv(scores_output_path, index=False, encoding='utf-8')
print(f"\nAll model cross-validation scores saved to: {scores_output_path}")

print("\n--- Evaluation Complete ---")