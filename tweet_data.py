import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import os
import nltk
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.externals import joblib

TEXT_EXT = '.text'
LABELS_EXT = '.labels'
UNK_SYMBOL = "<UNK>"
PAD_SYMBOL = "<PAD>"

class TweetsBaseDataset(data.Dataset):
    """ A Dataset class for the emoji prediction task. The base class reads
    the file to prepare a vocabulary that is then used for specialized
    subclasses.
    Args:
        - path (str): path to folder containing files
        - prefix (str): prefix of text and label files to load
        - vocab_size (int): maximum number of unique tokens to index
        - vocabulary (dict): maps tokens (str) to indices (int). If provided
            it is used instead of building it from the dataset, and
            vocab_size is ignored.
    """
    def __init__(self, path, prefix, vocab_size=10000, vocabulary=None):
        self.prefix = prefix
        token_counts = Counter()
        processed_tweets = []
        self.length = 0

        # Open text file with tweets
        print('Reading file')
        with open(os.path.join(path, prefix + TEXT_EXT)) as file:
            for i, line in enumerate(file):
                self.length += 1
                # Tokenize and process line
                tokens = self.process_tweet(line)
                processed_tweets.append(tokens)

                if vocabulary is None:
                    token_counts.update(tokens)

        print('Read file with {:d} tweets'.format(self.length))

        # Build vocabulary
        if vocabulary is None:
            print('Building vocabulary')
            vocabulary = defaultdict(lambda: len(vocabulary))
            _ = vocabulary[PAD_SYMBOL]
            _ = vocabulary[UNK_SYMBOL]
            for token, _ in token_counts.most_common(vocab_size):
                _ = vocabulary[token]
        else:
            print('Using vocabulary containing {:d} tokens'.format(
                len(vocabulary)))


        self.vocabulary = dict(vocabulary)

        # Store text in memory as word ids
        self.text_ids = []
        for tweet in processed_tweets:
            self.text_ids.append(list(map(
                lambda x: self.vocabulary.get(x, self.vocabulary[UNK_SYMBOL]),
                tweet)))

        # Load labels
        print('Loading labels')
        self.labels = np.empty(self.length, dtype=np.int)
        with open(os.path.join(path, prefix + LABELS_EXT)) as file:
            for i, line in enumerate(file):
                self.labels[i] = int(line)

    def __getitem__(self, index):
        return (torch.tensor(self.text_ids[index], dtype=torch.long),
                torch.tensor(self.labels[index], dtype=torch.long))

    def __len__(self):
        return self.length

    def process_tweet(self, text):
        """ Process and tokenize a tweet.
        Args:
            - text (str): a raw tweet in string format
        Returns: list, containing tokens after processing
        """
        # More operations can be added here before returning list of tokens
        return nltk.word_tokenize(text)

    @staticmethod
    def collate_fn(data_list, batch_first=False):
        """
        Prepare a batch from a list of samples.
        Args:
            - data_list (list): contains tuples, each with two tensors
                as returned by __getitem__() in the Dataset class.
        Returns:
            - packed_data (tensor): padded sequences forming a batch
            - labels (tensor): batch of labels
        """
        # Separate token indices and labels
        data, labels = zip(*data_list)

        # Get length of each tensor
        lengths = np.array([len(tensor) for tensor in data])
        # Sort tensors from longest to shortest
        sorted_idx = np.argsort(lengths)[::-1]
        sorted_data = [data[idx] for idx in sorted_idx]
        sorted_labels = torch.stack([labels[idx] for idx in sorted_idx])
        sorted_lengths = lengths[sorted_idx]

        # Create padded batch
        padded_data = pad_sequence(sorted_data, batch_first)

        return padded_data, sorted_labels, sorted_lengths

class TweetsBOWDataset(TweetsBaseDataset):
    """ A Dataset class for the emoji prediction task that stores tweets as
        bag of words.
    Args:
        - path (str): path to folder containing files
        - prefix (str): prefix of text and label files to load
        - vocab_size (int): maximum number of unique words to index
        - vocabulary (dict): maps tokens (str) to indices (int). If provided
            it is used instead of building it from the dataset, and
            vocab_size is ignored.
    """
    def __init__(self, path, prefix, vocab_size=10000, vocabulary=None):
        TweetsBaseDataset.__init__(self, path, prefix, vocab_size, vocabulary)

        # Using the vocabulary, build count matrix from text ids
        print('Loading counts matrix')
        count_matrix = lil_matrix((self.length, len(self.vocabulary)),
                                  dtype=np.int)
        for i, token_ids in enumerate(self.text_ids):
            # Count ids in tweet
            token_id_counter = Counter(token_ids)
            token_ids, counts = zip(*token_id_counter.items())
            count_matrix[i, token_ids] = counts

        # Add tf-idf weighting
        print('Creating TF-ID matrix')
        self.data = TfidfTransformer().fit_transform(count_matrix)

if __name__ == '__main__':
    ds = TweetsBOWDataset('data/dev', 'us_trial')