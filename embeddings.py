import os
import numpy as np
from tweet_data import TweetsBaseDataset


def get_embeddings(filename, vocabulary, dim=300):
    """
    Load pretrained embeddings of words in a vocabulary from a text file.
    Args:
        - filename (str): text file containing embeddings, each line containing
            the word and the embedding values separated by spaces.
        - vocabulary (dict): maps tokens (str) to indices (int).
        - dim (int): dimensionality of the embeddings to load.
    Returns: embeddings, numpy array of shape (len(vocabulary), emb_dim) where
        emb_dim is the dimensionality of the loaded embeddings. The embedding
        of word w is stored in the i-th row of embeddings, where
        i = vocabulary[w].
    """
    # Initialize embeddings from standard normal
    embeddings = np.random.randn(len(vocabulary), dim).astype(np.float32)
    words_found = 0

    with open(filename) as file:
        for i, line in enumerate(file):
            # Get word and embedding values
            values = line.strip().split()
            word = values[0]
            # Store embedding only if in vocabulary
            if word in vocabulary:
                words_found += 1
                word_idx = vocabulary[word]
                embedding = np.array(values[1:]).astype(np.float)
                try:
                    embeddings[word_idx] = embedding
                except ValueError:
                    raise ValueError(
                        'Read embedding of length {:d}, expected {:d}'.format(
                        len(embedding), dim))

    print('Loaded {:d} embeddings out of {:d} words in vocabulary'.format(
        words_found, len(vocabulary)))

    return embeddings

if __name__ == '__main__':
    # When run as a script embeddings are loaded and serialized,
    # given a vocabulary in an existing training set
    data_dir = './data'
    embeddings_dir = './embeddings'

    train_set = TweetsBaseDataset.load(
        os.path.join(data_dir, 'train', 'us_train.set'))

    embeddings = get_embeddings(os.path.join(embeddings_dir,
                    'ntua_twitter_300.txt'), train_set.vocabulary)

    embeddings_fname = 'embeddings.npy'
    np.save(os.path.join(embeddings_dir, embeddings_fname), embeddings)
    print('Saved embeddings to {}'.format(embeddings_fname))
