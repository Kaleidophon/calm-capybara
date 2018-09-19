"""
Module that defines some baseline models for the emoji prediction task.
"""

# EXT
from sklearn import linear_model
from sklearn.metrics import f1_score, precision_score, recall_score

# PROJECT
from tweet_data import TweetsBOWDataset


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
            "penalty": "l2", "max_iter": 10, "random_state": 42
        },
        "svm": {
            "loss": "hinge", "penalty": "l2", "alpha": 1e-3, "random_state": 42, "max_iter": 10, "tol": None
        },
    }

    def __init__(self, classifier="logistic_regression"):
        """
        Initialize the model.

        Args:
            - classifier (str): Classifier to choose from. One of {"logistic_regression", "svm"}.
        """
        assert classifier in self.scikit_models, "Invalid classifier. Pick one of {}".format(
            " ,".join(list(self.scikit_models.keys()))
        )

        self.model = self.scikit_models[classifier](**self.scikit_params[classifier])

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
        """
        predictions = self.model.predict(X=dataset.data)

        # Get scores
        precision = precision_score(dataset.labels, predictions, average="macro")
        recall = recall_score(dataset.labels, predictions, average="macro")
        f1 = f1_score(dataset.labels, predictions, average="macro")

        return precision, recall, f1


if __name__ == "__main__":
    # Load data sets
    english_train = TweetsBOWDataset("data/train", "us_train")
    english_test = TweetsBOWDataset("data/test", "us_test")

    # Train models
    bow_baseline = BoWBaseline(classifier="svm")
    bow_baseline.train(english_test)
    p, r, f1 = bow_baseline.eval(english_test)
    print("Precision: {:.2f} | Recall: {:.2f} | F1-Score: {:.2f}".format(p, r, f1))
