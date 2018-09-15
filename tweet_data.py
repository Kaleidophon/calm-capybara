import torch
from torch.utils import data
import os
import nltk
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from scipy.sparse import lil_matrix

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
        - vocab_size (int): maximum number of unique words to index
    """
    def __init__(self, path, prefix, vocab_size=10000):
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
                token_counts.update(tokens)
                processed_tweets.append(tokens)

        print('Read file with {:d} tweets, {:d} unique tokens'.format(
            self.length, len(token_counts)))

        # Build vocabulary
        print('Building vocabulary')
        vocabulary = defaultdict(lambda: len(vocabulary))
        _ = vocabulary[PAD_SYMBOL]
        unk_idx = vocabulary[UNK_SYMBOL]
        for token, _ in token_counts.most_common(vocab_size):
            _ = vocabulary[token]
        self.vocabulary = dict(vocabulary)

        # Store text in memory as word ids
        self.text_ids = []
        for tweet in processed_tweets:
            self.text_ids.append(
                list(map(lambda x: self.vocabulary.get(x, unk_idx), tweet)))

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


class TweetsBOWDataset(TweetsBaseDataset):
    """ A Dataset class for the emoji prediction task that stores tweets as
        bag of words.
    Args:
        - path (str): path to folder containing files
        - prefix (str): prefix of text and label files to load
        - vocab_size (int): maximum number of unique words to index
    """
    def __init__(self, path, prefix, vocab_size=10000):
        TweetsBaseDataset.__init__(self, path, prefix, vocab_size)

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
