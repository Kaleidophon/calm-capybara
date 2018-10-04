"""
Define the convolutional neural network for emoji prediction.
"""
# STD
import os

# EXT
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# PROJECT
from train import train_model
from tweet_data import TweetsBaseDataset


class CNNClassifier(nn.Module):

    def __init__(self, embeddings, num_kernels=3, n_classes=20, dropout=0.5, use_pretrained_embeddings=False,
                 train_embeddings=False):
        super().__init__()

        # Initialize embeddings - take pre-trained ones (or not) and determine whether they should be trained further
        num_embeddings, embedding_dim = embeddings.shape

        if use_pretrained_embeddings:
            self.embeddings = nn.Embedding(num_embeddings, embedding_dim, _weight=torch.from_numpy(embeddings))
        else:
            self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.requires_grad = train_embeddings

        # Initialize convolutional layers
        self.conv_size3 = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=(3, embedding_dim))
        self.conv_size4 = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=(4, embedding_dim))
        self.conv_size5 = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=(5, embedding_dim))
        self.filters = nn.ModuleList([self.conv_size3, self.conv_size4, self.conv_size5])

        # Define linear layer with droput
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(3 * num_kernels, n_classes)

    def forward(self, input, *args):
        # b: Batch size
        # l: Sequence length
        # d: embedding dim
        # ch_in: Channels in
        # ch_out: Channels out
        # k: kernel size
        # c: Number of classes
        input = input.t()  # Transpose because the batches in the format l x b

        x = self.embeddings(input)  # b x l x d
        x = x.unsqueeze(1)  # b x 1 x ch_in x l x d

        # Apply kernels
        # List of tensors b x ch_out x (l - k + 1) x 1
        feature_maps = [kernel(x) for kernel in self.filters]

        # Apply activation function
        # List of tensors b x ch_out x (l - k + 1) x 1
        feature_maps = [F.relu(fm).squeeze(3) for fm in feature_maps]

        # Perform max-over-time pooling
        # List of tensors b x ch_out
        feature_maps = [F.max_pool1d(fm, fm.size(2)).squeeze(2) for fm in feature_maps]
        # b x ch_out * 3
        c = torch.cat(feature_maps, 1)

        # Run through linear layer
        # Output b x c
        c = self.drop(c)
        out = self.linear(c)

        return out
# num filters
# pretrained or not
# num linear layers
# dropout

if __name__ == "__main__":
    # Load data sets~
    root_dir = "."
    english_train = TweetsBaseDataset.load(root_dir + "/data/train/us_train.set")
    english_dev = TweetsBaseDataset.load(root_dir + "/data/dev/us_trial.set")
    english_test = TweetsBaseDataset.load(root_dir + "/data/test/us_test.set")
    datasets = (english_train, english_dev, english_test)

    embeddings_dir = root_dir + '/embeddings'
    embeddings = np.load(os.path.join(embeddings_dir, 'embeddings.npy'))

    # Init model and begin training
    model = CNNClassifier(embeddings, train_embeddings=True, use_pretrained_embeddings=True, num_kernels=12)
    metadata = {
        "Model name": "Best CNN", "embeddings": "Train from scratch", "Num filters": 12, "Num linear": 1,
        "Regularization": "Dropout"
    }
    print(metadata)

    #torch.manual_seed(42)
    train_model(model, datasets, batch_size=256, epochs=60, learning_rate=1e-3, metadata=metadata)
