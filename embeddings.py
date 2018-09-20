import numpy as np
from tweet_data import TweetsBaseDataset
from torch.nn import Embedding

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
    embeddings = np.random.randn(len(vocabulary), dim)
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
                embeddings[word_idx] = np.array(values[1:]).astype(np.float)

    print('Loaded {:d} embeddings out of {:d} words in vocabulary'.format(
        words_found, len(vocabulary)))

    return embeddings
