# Data processing
from tweet_data import*

# STD
import itertools
import operator
import functools

# EXT
import fasttext as ft
from sklearn.metrics import f1_score, precision_score, recall_score


def preprocess_fasttext(path, prefix):
    """
    Prepares dataset s.t. it is ready to use for the fastText classifier.
    The format of each line in the input file is the label for each line
    of text followed by the corresponding tweet. Example:

    __label__15,Views from the Bay club @ Bayside Bay Club

    Args:
        - path (str): path to folder containing files
        - prefix (str): prefix of text and label files to load
    """

    TEXT_EXT = '.text'
    LABELS_EXT = '.labels'
    label_prefix = '__label__'

    data_fasttext_format = []
    length = 0

    # Open text file containing tweets
    print('Reading file')
    with open(os.path.join(path, prefix + TEXT_EXT)) as file:
        for i, line in enumerate(file):
            length += 1
            data_fasttext_format.append([line])

    print('Read file with {:d} tweets'.format(length))

    # Write formatted data to file
    with open(os.path.join(path, 'ft_' + prefix + TEXT_EXT), 'w+') as out_file:
        print('Loading labels & reformatting text')
        # Open corresponding label file
        with open(os.path.join(path, prefix + LABELS_EXT)) as file:
            for i, line in enumerate(file):
                new_label = label_prefix + str((int(line)))
                data_fasttext_format[i].insert(0, new_label)
                tweet = data_fasttext_format[i][0] + str(' ')+data_fasttext_format[i][1]
                out_file.write(tweet)

def F1_score(precision, recall):
    f1_score = 2 * ((precision * recall)/(precision + recall))
    return f1_score

def eval_model(model, DATA_FOLDER, FILE,TEXT_EXT):
    '''
    Evaluating the model using precision, recall and F-1 score.

    Args:
        - model: trained classifier
        - test_data: path to test data
    Returns:
        - precison (float): Precision score @1.
        - recall (float): Recall score @1.
        - f1score (float): F1 score.
    '''
    LABELS_EXT = '.labels'
    pred_labels = []
    with open(DATA_FOLDER+'us_trial'+TEXT_EXT) as file:
        for line in file:
            pred_labels.append(int(model.predict(line)[0][0]))

    # Load labels
    label_file = DATA_FOLDER+'us_trial'+LABELS_EXT
    target_labels = []
    with open(os.path.join(label_file)) as file:
        for i, line in enumerate(file):
            target_labels.append(int(line))

    # Calculate evaluation metrics
    precision = precision_score(target_labels, pred_labels, average="macro")
    recall = recall_score(target_labels, pred_labels, average="macro")
    f1score = f1_score(target_labels, pred_labels, average="macro")

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1score)

    return precision, recall, f1score

def grid_search(hyperparameter_options, TRAIN_FOLDER, TRAIN_FILE, TEST_FOLDER, TEST_FILE, DEV_FOLDER, DEV_FILE):

    TEXT_EXT = '.text'
    MODEL_FILE = 'data/ft_BEST_model'

    def _train_and_eval(model_params, MODEL_FILE, DATA_FOLDER, FILE, TEXT_EXT):
        current_model = train_ft(TRAIN_FOLDER + TRAIN_FILE + TEXT_EXT, MODEL_FILE, **model_params)
        return eval_model(current_model, DATA_FOLDER, FILE, TEXT_EXT)

    highest_score = -1
    p_best, r_best = 0, 0
    best_parameters = None

    print("Running grid search with options {}".format(str(hyperparameter_options)))

    n_combinations = functools.reduce(operator.mul, [len(options) for options in hyperparameter_options.values()])
    for i, hyperparams in enumerate(itertools.product(*hyperparameter_options.values())):
        current_model_params = dict(zip(hyperparameter_options.keys(), hyperparams))

        print(
            "\rTrying out combination {}/{}: {}".format(
                i + 1, n_combinations, str(current_model_params)
            ), flush=True, end=""
        )

        p, r, f1 = _train_and_eval(current_model_params, MODEL_FILE, DEV_FOLDER, DEV_FILE, TEXT_EXT)

        print("\nPrecision: {:.4f} | Recall: {:.4f} | F1-score: {:.4f}".format(p, r, f1))

        if f1 > highest_score:
            print("New highest score found ({:.4f})".format(f1))
            p_best, r_best, highest_score = p, r, f1
            best_parameters = current_model_params

    # Obtain test set performance for best params
    p_test, r_test, f1_test = _train_and_eval(best_parameters, MODEL_FILE, TEST_FOLDER, TEST_FILE, TEXT_EXT)

    print("Found best parameters")
    print(str(best_parameters))
    print("achieving the following performances:")
    print("Dev set: Precision: {:.4f} | Recall: {:.4f} | F1-score: {:.4f}".format(p_best, r_best, highest_score))
    print("Test set: Precision: {:.4f} | Recall: {:.4f} | F1-score: {:.4f}\n".format(p_test, r_test, f1_test))

    return best_parameters

def train_ft(TRAIN_FILE, MODEL_FILE,loss, lr, dim, epoch, ws):#, silent):
    '''
    Trains a baseline text classifier as in Joulin et al. 2017 using a Skip-gram Negative Sampling
    loss function.
    Further documentation: github.com/facebookresearch/fastText
                            https://github.com/facebookresearch/fastText#full-documentation


    Args:
        -TRAIN_FILE: path to training file
        -MODEL_FILE: path to file in which trained model will be saved
        - lr: learning rate
        - dim: size of word vectors
        - epoch: number of epochs
        - ws: size of context window
        - silent: if set to 0, prints training information to stdout

    Returns:
        - classifier: trained fastText classifier
    '''
    classifier = ft.supervised(TRAIN_FILE, MODEL_FILE, loss=loss, lr=lr, dim=dim, epoch=epoch, ws=ws)#, silent=silent)
    return classifier

if __name__ == "__main__":
    np.random.seed(42)

    #preprocess_fasttext("data/test", "us_test")
    #preprocess_fasttext("data/test", "es_test")
    #preprocess_fasttext("data/dev","us_trial")
    #preprocess_fasttext("data/dev", "es_trial")
    #preprocess_fasttext("data/train", "us_train")
    #preprocess_fasttext("data/train", "es_train")

    MODEL_FILE = 'data/fasttext_model'
    TRAIN_FOLDER = 'data/train/'
    TRAIN_FILE = 'ft_us_train'
    TEST_FOLDER = 'data/test/'
    TEST_FILE = 'ft_us_test'
    DEV_FOLDER = 'data/dev/'
    DEV_FILE = 'ft_us_trial'

    # Perform grid search to find best hyperparameters (ref:http://soner.in/fasttext-grid-search/)
    hyperparameter_options_ft = {
        "epoch": [50, 100],
        "lr": [0.10, 0.05, 0.01],
        "loss": ["ns", "softmax"],
        "dim": [300, 100],
        "ws": [5, 10, 25],
        #"silent": [0]
    }

    best = grid_search(hyperparameter_options_ft, TRAIN_FOLDER, TRAIN_FILE, TEST_FOLDER, TEST_FILE,  DEV_FOLDER, DEV_FILE)
    print(best)


