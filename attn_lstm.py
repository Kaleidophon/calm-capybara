import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from train import train_model
from tweet_data import TweetsBaseDataset
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, output, hidden):
        hidden = hidden.squeeze(0)
        attn_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)
        attention = F.softmax(attn_weights, 1)
        context = torch.bmm(output.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        return context


class AttentionLSTMClassifier(nn.Module):
    def __init__(self, embeddings, n_classes=20, dropout=0):
        super(AttentionLSTMClassifier, self).__init__()
        num_embeddings, embedding_dim = embeddings.shape
        self.embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim,
                                       _weight=torch.from_numpy(embeddings))
        self.lstm = nn.LSTM(embedding_dim, embedding_dim)
        self.attention = Attention()
        self.linear = nn.Linear(embedding_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        x = self.embeddings(inputs)
        x = pack_padded_sequence(x, lengths)
        output, (hidden, cell) = self.lstm(x)
        unpacked, unpacked_len = pad_packed_sequence(output)
        output = unpacked.permute(1, 0, 2)

        attn_output = self.attention(output, hidden)

        # First dimension is sequence dimension which here always equals 1
        # so we remove it before passing it to the linear layer
        logits = self.dropout(self.linear(attn_output))

        return logits

if __name__ == '__main__':
    embeddings_dir = './embeddings'

    embeddings = np.load(os.path.join(embeddings_dir, 'embeddings.npy'))
    model = AttentionLSTMClassifier(embeddings)

    data_dir = './data'
    train_set = TweetsBaseDataset.load(os.path.join(data_dir, 'train',
            'us_train.set'))
    dev_set = TweetsBaseDataset.load(os.path.join(data_dir, 'dev',
            'us_trial.set'))
    test_set = TweetsBaseDataset.load(os.path.join(data_dir, 'test',
            'us_test.set'))

    datasets = (train_set, dev_set, test_set)

    metadata = {'Model name': 'basic LSTM'}
    train_model(model, datasets, batch_size=32, epochs=20,
                learning_rate=1e-3, metadata=metadata)
