from torch.utils import data
import os
import nltk
from collections import Counter, defaultdict

TEXT_EXT = '.text'
LABELS_EXT = '.labels'
UNK_SYMBOL = "<UNK>"
PAD_SYMBOL = "<PAD>"

class TweetsBaseDataset(data.Dataset):
    """ A Dataset class for the emoji prediction task
    Args:
        - path (str): path to folder containing files
        - prefix (str): prefix of text and label files to load
        - vocab_size (int): maximum number of unique words to index
    """
    def __init__(self, path, prefix, vocab_size=100):
        self.prefix = prefix
        token_counts = Counter()
        processed_tweets = []

        # Open text file with tweets
        with open(os.path.join(path, prefix + TEXT_EXT)) as file:
            for i, line in enumerate(file):
                # Tokenize and process line
                tokens = self.process_tweet(line)
                token_counts.update(tokens)
                processed_tweets.append(tokens)

        # Build vocabulary and store words as integers
        self.vocabulary = defaultdict(lambda: len(self.vocabulary))
        _ = self.vocabulary[PAD_SYMBOL]
        unk_idx = self.vocabulary[UNK_SYMBOL]
        for token, _ in token_counts.most_common(vocab_size):
            _ = self.vocabulary[token]
        self.vocabulary.default_factory = lambda: unk_idx

    def __getitem__(self, index):
        pass

    def process_tweet(self, text):
        """ Process and tokenize a tweet.
        Args:
            - text (str): a raw tweet in string format
        Returns: list, containing tokens after processing
        """
        return nltk.word_tokenize(text)


if __name__ == '__main__':
    ds = TweetsBaseDataset('data/dev', 'es_trial')
