import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from train import train_model
from tweet_data import TweetsBaseDataset
import itertools

class biLSTMClassifier(nn.Module):
    def __init__(self, embeddings, n_classes=20, emb_dropout=0.0,
                 lstm_dropout=0.0):
        super(biLSTMClassifier, self).__init__()

        num_embeddings, embedding_dim = embeddings.shape
        self.embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim,
                                       _weight=torch.from_numpy(embeddings))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(embedding_dim * 2, n_classes)

    def forward(self, inputs, lengths):
        x = self.embeddings(inputs)
        x = self.emb_dropout(x)
        x = pack_padded_sequence(x, lengths)
        _, (hidden, cell) = self.lstm(x)
        x = torch.cat((hidden[0], hidden[1]), dim=1)
        x = self.lstm_dropout(x)
        logits = self.linear(x)

        return logits

if __name__ == '__main__':
    # Load pretrained embeddings
    embeddings_dir = './embeddings'
    embeddings = np.load(os.path.join(embeddings_dir, 'embeddings.npy'))

    # Load task data
    data_dir = './data'
    train_set = TweetsBaseDataset.load(os.path.join(data_dir, 'train',
            'us_train.set'))
    dev_set = TweetsBaseDataset.load(os.path.join(data_dir, 'dev',
            'us_trial.set'))
    test_set = TweetsBaseDataset.load(os.path.join(data_dir, 'test',
            'us_test.set'))
    datasets = (train_set, dev_set, test_set)

    # Hyperparameter search
    emb_dropout = [0.0, 0.1, 0.3, 0.5]
    lstm_dropout = [0.0, 0.1, 0.3, 0.5]
    learning_rate = [1e-2, 1e-3, 1e-4]

    best_val_score = 0
    best_test_score = 0
    best_hparams = {'emb_dropout': 0.0,
                    'lstm_dropout': 0.0,
                    'learning_rate': 1e-2}

    for ed, ld, lr in itertools.product(emb_dropout, lstm_dropout, learning_rate):
        model = biLSTMClassifier(embeddings, emb_dropout=ed, lstm_dropout=ld)

        metadata = {'Model name': 'BiLSTM',
                    'Embeddings dropout': ed,
                    'LSTM dropout': ld}

        val_score, test_score = train_model(model, datasets, batch_size=32,
            epochs=20, learning_rate=lr, metadata=metadata,
            weights={'linear': model.linear.weight.data})

        if val_score > best_val_score:
            best_val_score = val_score
            best_test_score = test_score
            best_hparams['emb_dropout'] = ed
            best_hparams['lstm_dropout'] = ld
            best_hparams['learning_rate'] = lr

    print('Best validation score: {:.4f}'.format(best_val_score))
    print('Test score: {:.4f}'.format(best_test_score))
    print('Best hyperparameters:\n{}'.format(best_hparams))
