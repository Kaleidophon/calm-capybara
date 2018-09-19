"""
Module that defines some baseline models for the emoji prediction task.
"""

# STD
import itertools
import operator
import functools

# EXT
from sklearn import linear_model
from sklearn.metrics import f1_score, precision_score, recall_score

# PROJECT
from tweet_data import TweetsBOWDataset, TweetsBaseDataset


class BoWBaseline:
    """
    Defining the Bag-of-Words baseline, using logistic regression or an SVM in order to predict the target emoji.
    """
    scikit_models = {
        "logistic_regression": linear_model.LogisticRegression,
        "svm": linear_model.SGDClassifier
    }
    scikit_params = {
        "logistic_regression": {
            'classifier': 'logistic_regression', 'max_iter': 30, 'penalty': 'l1', 'random_state': 42, 'tol': 0.1,
            'solver': 'saga'
        },
        "svm": {
            'classifier': 'svm', 'max_iter': 30, 'penalty': 'l1', 'random_state': 42, 'alpha': 1e-05, 'tol': 0.01,
            'loss': 'log'
        },
    }

    def __init__(self, classifier="logistic_regression", **model_params):
        """
        Initialize the model.

        Args:
            - classifier (str): Classifier to choose from. One of {"logistic_regression", "svm"}.
        """
        assert classifier in self.scikit_models, "Invalid classifier. Pick one of {}".format(
            " ,".join(list(self.scikit_models.keys()))
        )

        model_params = self.scikit_params[classifier] if len(model_params) == 0 else model_params
        self.model = self.scikit_models[classifier](**model_params)

    def train(self, dataset: TweetsBOWDataset):
        """
        Training the model using a TweetsBOWDataset.

        Args:
            - dataset (TweetsBOWDataset): Dataset to train on.
        """
        self.model.fit(X=dataset.data, y=dataset.labels)

    def eval(self, dataset: TweetsBOWDataset):
        """
        Evaluating the model using precision, accuracy and F1-score.

        Args:
            - dataset (TweetsBOWDataset): Dataset to evaluate the model on.

        Returns:
            - precision (float): Precision score.
            - recall (float): Recall score.
            - f1 (float): F1 score.
        """
        predictions = self.model.predict(X=dataset.data)

        # Get scores
        precision = precision_score(dataset.labels, predictions, average="macro")
        recall = recall_score(dataset.labels, predictions, average="macro")
        f1 = f1_score(dataset.labels, predictions, average="macro")

        return precision, recall, f1


def grid_search(model_class, train_set: TweetsBaseDataset, test_set: TweetsBaseDataset, hyperparameter_options: dict):
    """
    Perform grid search in order to find the best parameters for a model.

    Args:
        - model_class (type): Class for which the best hyperparameters should be determined.
        - train_set (TweetsBaseDataset): Dataset for the model to be trained on.
        - test_set (TweetsBaseDataset): Dataset for the model to be evaluated on.
        - hyperparameter_options (dict): Dictionary of hyperparameter to possible options (str -> list).
    """
    highest_score = -1
    best_parameters = None
    print("Trying to find best model parameters with options: {}".format(str(hyperparameter_options)))

    # Perform grid search
    n_combinations = functools.reduce(operator.mul, [len(options) for options in hyperparameter_options.values()])
    for i, hyperparams in enumerate(itertools.product(*hyperparameter_options.values())):
        current_model_params = dict(zip(hyperparameter_options.keys(), hyperparams))

        print(
            "\rTrying out combination {}/{}: {}".format(
                i + 1, n_combinations, str(current_model_params)
            ), flush=True, end=""
        )

        current_model = model_class(**current_model_params)
        current_model.train(train_set)
        p, r, f1 = current_model.eval(test_set)
        print("\nPrecision: {:.4f} | Recall: {:.4f} | F1-score: {:.4f}".format(p, r, f1))

        if f1 > highest_score:
            print("New highest score found ({:.4f})".format(f1))
            highest_score = f1
            best_parameters = current_model_params

    print("Best parameters with a score of {:.4f} where {}".format(highest_score, str(best_parameters)))
    return best_parameters


if __name__ == "__main__":
    # Load data sets
    # english_train = TweetsBOWDataset("data/train", "us_train")
    # english_train.dump("data/us_train.set")
    # english_test = TweetsBOWDataset("data/test", "us_test", vocabulary=english_train.vocabulary)
    # english_test.dump("data/us_test.set")

    english_train = TweetsBOWDataset.load("data/us_train.set")
    english_test = TweetsBOWDataset.load("data/us_test.set")

    # Train models and find best hyperparameters
    hyperparameter_options_svm = {
        "classifier": ["svm"],
        "max_iter": [30],
        "penalty": ["l1", "l2"],
        "random_state": [42],
        "alpha": [0.00001, 0.0001, 0.001, 0.01],
        "tol": [None, 0.0001, 0.001, 0.01],
        "loss": ["hinge", "log", "squared_hinge"],
        "n_jobs": [4]
    }
    hyperparameter_options_lr = {
        "classifier": ["logistic_regression"],
        "max_iter": [30],
        "penalty": ["l1", "l2"],
        "random_state": [42],
        "tol": [0.0001, 0.001, 0.01, 0.1],
        "solver": ["liblinear", "saga"],
        "n_jobs": [1]
    }
    grid_search(BoWBaseline, english_train, english_test, hyperparameter_options_svm)
    grid_search(BoWBaseline, english_train, english_test, hyperparameter_options_lr)
