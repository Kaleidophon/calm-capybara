"""
Training and evaluation functions for the emoji prediction task
"""

import numpy as np
import os

# Torch modules
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Data loading modules
from tweet_data import TweetsBOWDataset, TweetsBaseDataset

from sklearn.metrics import accuracy_score, f1_score

# Check for CUDA device / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory in which tweet data is saved
DATA_DIR_DEFAULT = './data'
TEST_BATCH_SIZE = 128

def get_score(logits, targets, score='f1_score'):
    """
    Computes the score of the network.
    Args:
        - logits (tensor): predictions of the model.
            shape (n_batches, n_classes)
        - targets (tensor): true labels, containing values in [0, n_classes-1]
            shape (n_batches,)
        - score (str): one of 'accuracy', 'f1_score'
    Returns: float, the calculated score.
    """
    predictions = torch.argmax(logits, dim=1)

    if score == 'accuracy':
        return accuracy_score(targets.data.numpy(), predictions.data.numpy())
    elif score == 'f1_score':
        return f1_score(targets.data.numpy(), predictions.data.numpy())

def evaluate(model, criterion, eval_data):
    # TODO: replace with continuous averaging
    test_loss = []
    accr = []

    # Load the test data
    data_loader = DataLoader(eval_data, collate_fn=TweetsBaseDataset.collate_fn,
                             batch_size=TEST_BATCH_SIZE, shuffle=True)

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(device)

            pred = model.forward(inputs)
            accr.append(get_score(pred, labels))

            labels = torch.LongTensor(labels).to(device)
            loss = criterion(pred, labels)
            test_loss.append(loss.item())

    return np.average(test_loss), np.average(accr)

def train_model(model, train_set, dev_set, batch_size, epochs, learning_rate,
                hparams=None, dataset_name='us'):

    # Set the seed for reproduction of results
    torch.manual_seed(123)

    train_loader = DataLoader(train_set, batch_size, shuffle=True,
                              num_workers=4,
                              collate_fn=TweetsBaseDataset.collate_fn)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        iteration = 1
        for inputs, labels, lengths in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Run the model
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            # Optimize
            loss.backward()
            optimizer.step()

            # Evaluate on training set
            accr = get_score(outputs, labels, score='accuracy')
            print("\rEpoch {}/{}: Training loss = {}, Training accuracy = {}".format(epoch,
                        iteration, "%.4f" % loss, "%.4f" % accr), end='', flush=True)

            iteration += 1

            # Evaluate on dev set
        eval_loss, eval_accr = evaluate(model, criterion, dev_set)
        print("\nValidation loss = {}, Validation accuracy = {}".format("%.4f" % eval_loss, "%.4f" % eval_accr))

    print("Training Completed")

    # Test the trained model
    #test_set = TweetsBaseDataset.load(os.path.join(DATA_DIR_DEFAULT,
    #    dataset_name + '_train.set'))
    #test_loss, test_accuracy = evaluate(model, criterion, test_set)
    #print("\nTest loss = {}, Test accuracy = {}".format("%.4f" % test_loss, "%.4f" % test_accuracy))
    #print("Test Completed")
