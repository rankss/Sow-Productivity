import numpy as np
from utils import create_directory, score_names
from graphing import plot_learning_curve

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, GridSearchCV, RandomizedSearchCV, learning_curve, cross_val_score, cross_val_predict
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.utils.validation import _deprecate_positional_args, check_random_state
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

# Adapted from https://github.com/scikit-learn/scikit-learn/issues/20317
class RepeatedGroupKFold(_RepeatedSplits):
    @_deprecate_positional_args
    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(GroupKFold, n_repeats=n_repeats, random_state=random_state, n_splits=n_splits)
        
    def split(self, X, y=None, groups=None):
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index
                
class EstimatorGlobalParameters:
    """Estimator parameters shared across classifiers.
    """
    
    def __init__(self, X, y, group, split_method, folds=5, repeats=10, random_state=None, directory=None, n_jobs=1):
        self.X = X
        self.y = y
        self.group = group
        self.split_method = split_method
        self.folds = folds
        self.repeats = repeats
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.directory = create_directory(directory)
 
    def train_test_split(self, random_state=None, holdout_group=1):
        """Splits data into training and testing sets based on the specified method.

        Args:
            random_state (int, optional): Seed for the random number generator.
                Used for 'random' split. Defaults to self.random_state.
            holdout_group (any, optional): The identifier for the group to be
                held out for testing in a 'logo' (Leave-One-Group-Out) split.

        Raises:
            ValueError: If an unsupported split_method is provided.
            ValueError: If split_method is 'logo' but holdout_group is not specified.

        Returns:
            tuple: A tuple containing (X_train, X_test, y_train, y_test, train_groups).
        """
        rng = random_state if random_state is not None else self.random_state

        if self.split_method == 'random':
            train_idx, test_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rng).split(
                    self.X, self.y, groups=self.group
                )
            )
            return self.X.iloc[train_idx], self.X.iloc[test_idx], self.y.iloc[train_idx], self.y.iloc[test_idx], self.group.iloc[train_idx]
        
        elif self.split_method == 'logo':
            train_idx = self.group.index[self.group != holdout_group]
            test_idx = self.group.index[self.group == holdout_group]
            return self.X.loc[train_idx], self.X.loc[test_idx], self.y.loc[train_idx], self.y.loc[test_idx], self.group.loc[train_idx]
            
        else:
            raise ValueError(f"Unknown split_method: '{self.split_method}'. "
                             "Supported methods are 'random' and 'logo'.")
    
    def cross_validator(self):
        """Returns a cross-validator object based on the specified split method.

        Raises:
            ValueError: If an unsupported split_method is provided.

        Returns:
            A scikit-learn compatible cross-validator instance (either
            RepeatedGroupKFold or LeaveOneGroupOut).
        """
        if self.split_method == 'random':
            return RepeatedGroupKFold(n_splits=self.folds, n_repeats=self.repeats, random_state=self.random_state)
        elif self.split_method == 'logo':
            return LeaveOneGroupOut()
        else:
            raise ValueError(f"Unknown split_method: '{self.split_method}'. "
                             "Supported methods are 'random' and 'logo'.")

class Estimator:
    """A wrapper for a scikit-learn estimator to streamline training, prediction,
    and hyperparameter tuning.
    """
    def __init__(self, estimator, name, params, grid=None):
        """Initializes the Estimator.

        Args:
            estimator: A scikit-learn compatible estimator instance.
            name: A string name for the estimator.
            params: An EstimatorGlobalParameters object with data and config.
            grid: A dictionary defining the hyperparameter grid for tuning.
                  If None, the estimator is considered pre-optimized.
        """
        self.estimator = estimator
        self.name = name
        self.params = params
        self.grid = grid
        self.optimization_state = "Optimized" if grid is None else "Default"
        
        # Split data once during initialization for consistent train/test sets
        self.X_train, self.X_test, self.y_train, self.y_test, self.train_group = self.params.train_test_split()
        self.cv = self.params.cross_validator()
        
    def fit(self, X_train=None, y_train=None):
        """Fits the estimator to the training data.

        Uses the instance's own training data if not provided.

        Args:
            X_train (pd.DataFrame, optional): The training feature data.
            y_train (pd.Series, optional): The training target data.

        Returns:
            The fitted Estimator instance.
        """
        if X_train is None or y_train is None:
            X_train = self.X_train
            y_train = self.y_train
            
        self.estimator.fit(X_train, y_train)
        return self
    
    def predict(self, X_test=None):
        """Makes predictions on a test set.

        Uses the instance's own test data if not provided.

        Args:
            X_test (pd.DataFrame, optional): The feature data to predict on.

        Returns:
            np.ndarray: An array of predictions.
        """
        if X_test is None:
            X_test = self.X_test
            
        return self.estimator.predict(X_test)
    
    def score(self, scoring_functions, X_test=None, y_test=None):
        """Scores the model's predictions against true labels.

        Uses the instance's own test data if not provided.

        Args:
            X_test (pd.DataFrame, optional): The feature data for scoring.
            y_test (pd.Series, optional): The true labels for scoring.
            scoring_functions (callable or dict): A scoring function or a dictionary
                of {name: scorer}, where scorer is a callable from sklearn.metrics.

        Returns:
            float or dict: The calculated score if a single function is provided,
                           or a dictionary of scores if a dictionary is provided.
        """
        if X_test is None or y_test is None:
            X_test = self.X_test
            y_test = self.y_test
            
        y_pred = self.predict(X_test)

        if callable(scoring_functions):
            return scoring_functions(y_test, y_pred)
        elif isinstance(scoring_functions, dict):
            scores = {}
            for name, scorer in scoring_functions.items():
                score = scorer(y_test, y_pred)
                scores[name] = score
            return scores

    def hyperparameter_optimization(self, scoring, search_method, n_iter=10, random_state=None):
        """Performs hyperparameter optimization and returns a new Estimator
        instance with the best found model.

        Args:
            scoring (str): The scoring metric to optimize for (e.g., 'accuracy', 'roc_auc').
            search_method (str): The search method, either 'grid' or 'random'.
            n_iter (int, optional): The number of iterations for random search. Defaults to 10.
            random_state (int, optional): The random seed for reproducibility in random search.

        Raises:
            ValueError: If the search method is unknown or if the estimator is
                        already marked as optimized.

        Returns:
            Estimator: A new instance initialized with the best found estimator.
        """
        if self.optimization_state == "Optimized":
            print("Estimator is already optimized. Skipping hyperparameter search.")
            self.fit() # Ensure the model is fitted on the current train/test split
            return self
        
        if search_method == 'grid':
            search = GridSearchCV(self.estimator, self.grid, scoring=scoring, cv=self.cv, n_jobs=self.params.n_jobs)
        elif search_method == 'random':
            try:
                n_iter = int(n_iter)
            except (ValueError, TypeError):
                raise ValueError(f"n_iter expects an integer, but got: {n_iter}")
            
            search = RandomizedSearchCV(
                self.estimator, self.grid, scoring=scoring, cv=self.cv, 
                n_iter=n_iter, random_state=random_state, n_jobs=self.params.n_jobs
            )
            
        else:
            raise ValueError(f"Unknown search_method: '{search_method}'. "
                             "Supported methods are 'grid' and 'random'.")
        
        print(f"Running {search_method} search for {self.name}...")
        search.fit(self.X_train, self.y_train, groups=self.train_group)
        
        print(f"Best score ({scoring}): {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        # Return a new instance of this class with the optimized estimator
        return self.__class__(
            estimator=search.best_estimator_, 
            name=self.name, 
            params=self.params, 
            grid=None  # Mark the new instance as optimized
        )
        
    def learning_curves(self, scoring, train_sizes=np.linspace(0.25, 1.0, 6)):
        """Generates and optionally saves learning curves for the estimator.

        Args:
            scoring (str): The scoring metric to use.
            train_sizes (array-like, optional): Relative or absolute numbers of training
                examples to use. Defaults to np.linspace(0.25, 1.0, 6).
            save (bool, optional): Whether to save the plot to a file. Defaults to True.

        Returns:
            matplotlib.figure.Figure: The figure object for the learning curve plot.
        """
        _, train_scores, test_scores = learning_curve(
            self.estimator, self.X_train, self.y_train, groups=self.train_group,
            cv=self.cv, scoring=scoring, train_sizes=train_sizes, n_jobs=self.params.n_jobs, 
            random_state=self.params.random_state
        )
        
        estimator_type = self.estimator._estimator_type.capitalize()
        plot_learning_curve(
            train_sizes, train_scores, test_scores, 
            f"{self.optimization_state} {self.name} {estimator_type}",
            xlabel="Training Set Size", ylabel=score_names.get(scoring, scoring),
            figure_directory=self.params.directory
        )
        
    def cross_validation(self, scoring):
        """Performs cross-validation and returns the scores.

        Args:
            scoring (str): The scoring metric to use.

        Returns:
            np.ndarray: An array of scores for each cross-validation split.
        """
        return cross_val_score(
            self.estimator, self.X_train, self.y_train, groups=self.train_group, 
            scoring=scoring, cv=self.cv, n_jobs=self.params.n_jobs
        )
    
    def cross_validation_predict(self):
        """Generates cross-validated estimates for each input data point.

        Returns:
            np.ndarray: An array of cross-validated predictions.
        """
        return cross_val_predict(
            self.estimator, self.X_train, self.y_train, groups=self.train_group, 
            cv=self.cv, n_jobs=self.params.n_jobs
        )
    
    def permutation_feature_importance(self, scoring, n_repeats=100, random_state=None):
        """Calculates permutation feature importance on the training set.

        Args:
            scoring (str): The scoring metric to use for evaluating feature importance.
            n_repeats (int, optional): Number of times to permute a feature. Defaults to 100.
            random_state (int, optional): Seed for the random number generator. Defaults to None,
                which then uses the global random state from `self.params`.

        Returns:
            object: A Bunch object with importance results.
        """
        rng = random_state if random_state is not None else self.params.random_state
        self.fit() # Ensure the model is fitted on the training data
        return permutation_importance(
            self.estimator, self.X_train, self.y_train, scoring=scoring,
            n_repeats=n_repeats, random_state=rng, n_jobs=self.params.n_jobs
        )
        
    def repeated_evaluation(self, scoring_functions, class_labels, repeats=100, random_state=None):
        """Evaluates the estimator multiple times with different random splits.

        Args:
            scoring_functions (callable or dict): A scoring function or a dictionary of
                {name: scorer}, where scorer is a callable from sklearn.metrics.
            repeats (int, optional): Number of evaluation repetitions. Defaults to 100.
            random_state (int, optional): Seed for the random number generator.
                Defaults to the global random state from `self.params`.
            class_labels (dict, optional): A dictionary mapping integer class codes
                to string labels (e.g., {0: 'Low', 1: 'Medium'}).

        Returns:
            dict: A dictionary of {scorer_name: [scores]} or
                  {scorer_name: {class_label: [scores]}}. Also includes a
                  'confusion_matrices' key with a list of confusion matrices
                  from each run.
        """
        # Allow scoring_functions to be a single callable
        if callable(scoring_functions):
            # Use the function's name as the key
            scoring_functions = {scoring_functions.__name__: scoring_functions}
        
        # Get integer class labels from the data
        class_labels_int = list(class_labels.keys())
        # Use the provided mapping for string labels, or default to integer strings
        class_labels_str = list(class_labels.values())

        # Initialize the results structure
        results = {}
        for name in scoring_functions.keys():
            results[name] = []  # Start with a simple list for all scorers
        results['confusion_matrices'] = []

        def evaluate(X_train, X_test, y_train, y_test):
            """Helper to run one evaluation iteration and append scores."""
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)

            # Store the confusion matrix for this run
            # Ensure confusion matrix has labels for consistent shape
            results['confusion_matrices'].append(confusion_matrix(y_test, y_pred, labels=class_labels_int))

            for name, scorer in scoring_functions.items():
                run_score = scorer(y_test, y_pred)

                # Check if the scorer returned per-class scores and if we are tracking them
                is_per_class = hasattr(run_score, '__iter__') and not isinstance(run_score, (str, bytes))

                if is_per_class:
                    # If this is the first time getting per-class scores, convert storage to a dict.
                    if isinstance(results[name], list):
                        results[name] = {label: [] for label in class_labels_str}
                    
                    # The scorer (e.g., f1_score with average=None and labels=...)
                    # returns scores in the same order as the provided labels.
                    for i, score in enumerate(run_score):
                        if i < len(class_labels_str):
                            results[name][class_labels_str[i]].append(score)
                else:
                    # If it's a single score, append it directly to the list.
                    results[name].append(run_score)

        if self.params.split_method == 'random':
            rng = random_state if random_state is not None else self.params.random_state
            rng = check_random_state(rng)
            seeds = rng.randint(np.iinfo(np.int32).max, size=repeats)

            for seed in seeds:
                X_train, X_test, y_train, y_test, _ = self.params.train_test_split(random_state=seed)
                if y_test.empty:
                    continue
                evaluate(X_train, X_test, y_train, y_test)

        elif self.params.split_method == 'logo':
            unique_groups = self.params.group.unique()
            for unique_group in unique_groups:
                X_train, X_test, y_train, y_test, _ = self.params.train_test_split(holdout_group=unique_group)
                if y_test.empty:
                    continue
                evaluate(X_train, X_test, y_train, y_test)

        else:
            raise ValueError(f"Unknown split_method: '{self.params.split_method}'. "
                             "Supported methods are 'random' and 'logo'.")

        # Post-processing to clean up empty lists from the results structure
        for name, value in list(results.items()): # Use list to allow modification during iteration
            if isinstance(value, dict):
                # Remove class labels that never received a score
                results[name] = {k: v for k, v in value.items() if v}

        return results