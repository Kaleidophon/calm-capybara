import itertools
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from train import train_model
from tweet_data import TweetsBaseDataset
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(2* input_size, 1)

    def forward(self, x, lengths):
        # Compute context vector as average of input sequence
        context = torch.sum(x, dim=1)/lengths.view(-1, 1).float()
        # Expand context so it can be concatenated with all input elements.
        # This creates copies of context along the sequence length axis
        context = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        inputs = torch.cat((x, context), dim=-1)
        # Get scores
        attention = F.softmax(self.linear(inputs).squeeze(), dim=-1)

        # Create mask to zero out contributions from padding
        col_lengths = lengths.unsqueeze(-1)
        len_matrix = torch.arange(lengths[0]).expand(len(lengths), -1).to(device)
        mask = (len_matrix < col_lengths).float().to(device).detach()
        attention = attention * mask

        # Renormalize
        attention = attention/torch.sum(attention, dim=-1, keepdim=True)
        # Return weighted average
        return torch.sum(x * attention.unsqueeze(-1), dim=1)

class AttentionBiLSTMClassifier(nn.Module):
    def __init__(self, embeddings, n_classes=20, emb_dropout=0.0,
                 lstm_dropout=0.0):
        super(AttentionBiLSTMClassifier, self).__init__()
        num_embeddings, embedding_dim = embeddings.shape
        self.embeddings = nn.Embedding(num_embeddings,
                                       embedding_dim,
                                       _weight=torch.from_numpy(embeddings))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.bilstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True)
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        self.attention = Attention(2 * embedding_dim)
        self.linear = nn.Linear(2 * embedding_dim, n_classes)

    def forward(self, inputs, lengths):
        # Get embeddings, including padding
        x = self.embeddings(inputs)
        x = self.emb_dropout(x)
        # Convert to PackedSequence so LSTM doesn't compute output on paddings
        x = pack_padded_sequence(x, lengths)
        # Get RNN (packed) sequence output
        x, _ = self.bilstm(x)
        # Convert back to padded sequence
        # discarding lengths as they are already in `lenghts`
        x, _ = pad_packed_sequence(x)
        x = self.lstm_dropout(x)
        # Move batch dimension to position 0
        # so that x.shape = (batch_size, max_seq_len, 2 * emb_dim)
        x = x.permute(1, 0, 2)
        # Use attention layer to get sentence representation
        x = self.attention(x, lengths)

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
    emb_dropout = [0.1, 0.3, 0.5]
    lstm_dropout = [0.1, 0.3, 0.5]
    learning_rate = [1e-3, 1e-4]

    best_val_score = 0
    best_test_score = 0
    best_hparams = {'emb_dropout': 0.0,
                    'lstm_dropout': 0.0,
                    'learning_rate': 1e-3}

    for ed, ld, lr in itertools.product(emb_dropout, lstm_dropout, learning_rate):
        model = AttentionBiLSTMClassifier(embeddings, emb_dropout=ed,
                                          lstm_dropout=ld)

        metadata = {'Model name': 'BiLSTM with attention',
                    'Embeddings dropout': ed,
                    'LSTM dropout': ld}

        val_score, test_score = train_model(model, datasets, batch_size=32,
            epochs=10, learning_rate=lr, metadata=metadata)

        if val_score > best_val_score:
            best_val_score = val_score
            best_test_score = test_score
            best_hparams['emb_dropout'] = ed
            best_hparams['lstm_dropout'] = ld
            best_hparams['learning_rate'] = lr

    print('Best validation score: {:.4f}'.format(best_val_score))
    print('Test score: {:.4f}'.format(best_test_score))
    print('Best hyperparameters:\n{}'.format(best_hparams))
