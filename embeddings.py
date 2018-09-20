import numpy as np

def get_embeddings(filename, vocabulary):
    """
    Load pretrained embeddings of words in a vocabulary from a text file.
    Args:
        - filename (str): text file containing embeddings, each line containing
            the word and the embedding values separated by spaces.
        - vocabulary (dict): maps tokens (str) to indices (int).
    Returns: embeddings, numpy array of shape (len(vocabulary), emb_dim) where
        emb_dim is the dimensionality of the loaded embeddings. The embedding
        of word w is stored in the i-th row of embeddings, where
        i = vocabulary[w].
    """
    # Read all embeddings from file
    word_to_embedding = {}
    with open(filename) as file:
        for i, line in enumerate(file):
            values = line.strip().split()
            word = values[0]
            embedding = np.array(values[1:]).astype(np.float)
            word_to_embedding[word] = embedding

    # Extract any embedding to get its dimensionality
    emb_dim = len(word_to_embedding[next(iter(word_to_embedding))])

    # Select embeddings needed by the vocabulary
    embeddings = np.zeros((len(vocabulary), emb_dim))
    for word in vocabulary:
        if word in word_to_embedding:
            word_idx = vocabulary[word]
            embeddings[word_idx] = word_to_embedding[word]

    return embeddings
