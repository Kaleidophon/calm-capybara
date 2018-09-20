import os

# Data processing
from tweet_data import*

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
        print('Loading labels')
        # Open corresponding label file
        with open(os.path.join(path, prefix + LABELS_EXT)) as file:
            for i, line in enumerate(file):
                new_label = label_prefix + str((int(line)))
                data_fasttext_format[i].insert(0, new_label)
                tweet = data_fasttext_format[i][0] + ','+data_fasttext_format[i][1]
                out_file.write(tweet)

if __name__ == "__main__":
    preprocess_fasttext("data/test", "us_test")
    preprocess_fasttext("data/test", "es_test")
    preprocess_fasttext("data/train", "us_train")
    preprocess_fasttext("data/train", "es_train")