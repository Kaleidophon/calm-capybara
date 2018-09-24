import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from train import train_model
from tweet_data import TweetsBaseDataset

class biLSTMClassifier(nn.Module):
    def __init__(self, embeddings, n_classes=20, dropout=0):
        super(biLSTMClassifier, self).__init__()
        num_embeddings, embedding_dim = embeddings.shape
        self.embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim,
                                       _weight=torch.from_numpy(embeddings))
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True)
        self.linear = nn.Linear(embedding_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        x = self.embeddings(inputs)
        x = pack_padded_sequence(x, lengths)
        _, (hidden, cell) = self.lstm(x)
        # Concat the outputs of the both direction LSTM before passing it to the next layer
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # First dimension is sequence dimension which here always equals 1
        # so we remove it before passing it to the linear layer
        logits = self.linear(hidden)

        return logits

if __name__ == '__main__':
    embeddings_dir = './embeddings'

    embeddings = np.load(os.path.join(embeddings_dir, 'embeddings.npy'))
    model = biLSTMClassifier(embeddings)

    data_dir = './data'
    train_set = TweetsBaseDataset.load(os.path.join(data_dir, 'train',
            'us_train.set'))
    dev_set = TweetsBaseDataset.load(os.path.join(data_dir, 'dev',
            'us_trial.set'))
    test_set = TweetsBaseDataset.load(os.path.join(data_dir, 'test',
            'us_test.set'))

    datasets = (train_set, dev_set, test_set)

    metadata = {'Model name': 'Bidirectional LSTM'}
    train_model(model, datasets, batch_size=32, epochs=20,
                learning_rate=1e-3, metadata=metadata)
