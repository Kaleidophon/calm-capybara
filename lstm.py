import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from train import train_model
from tweet_data import TweetsBaseDataset

class LSTMClassifier(nn.Module):
    def __init__(self, embeddings, n_classes=20):
        super(LSTMClassifier, self).__init__()
        num_embeddings, embedding_dim = embeddings.shape
        self.embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim,
                                       _weight=torch.from_numpy(embeddings))
        self.lstm = nn.LSTM(embedding_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, n_classes)

    def forward(self, inputs, lengths):
        x = self.embeddings(inputs)
        x = pack_padded_sequence(x, lengths)
        _, x = self.lstm(x)

        # x[0] contains hidden state
        # First dimension is sequence dimension which here always equals 1
        # So we want x[0][0] to remove the sequence dimension
        logits = self.linear(x[0][0])
        return logits

np.load('./embeddings/embeddings.npy')
embeddings = np.load('./embeddings/embeddings.npy')
model = LSTMClassifier(embeddings)

DATA_DIR_DEFAULT = './data'
train_set = TweetsBaseDataset.load(os.path.join(DATA_DIR_DEFAULT,
        'us_train.set'))
dev_set = TweetsBaseDataset.load(os.path.join(DATA_DIR_DEFAULT,
        'us_trial.set'))
test_set = TweetsBaseDataset.load(os.path.join(DATA_DIR_DEFAULT,
        'us_test.set'))

datasets = (train_set, dev_set, test_set)

train_model(model, datasets, batch_size=32, epochs=20,
            learning_rate=1e-3)