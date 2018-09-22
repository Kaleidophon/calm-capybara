"""
Neural network base model that includes the training and evaluation methods
for the emoji prediction task
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

# Check for CUDA device / GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory in which tweet data is saved
DATA_DIR_DEFAULT = './data'
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 12
TEST_BATCH_SIZE = 4


class Dummy_network(nn.Module):
    """
    This is a dummy class for the code testing purpose
    Replace this model with original NN model such as LSTM later
    """
    def __init__(self, *hyperparameters):
        super(Dummy_network, self).__init__()

        self.model = nn.Sequential()

    def forward(self, inputs):
        out = self.model(inputs)

        return out


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy of the network.
    """
    #pred_class = np.argmax(predictions, axis=1)
    #target_class = np.argmax(targets, axis=1)
    #accuracy = np.sum(pred_class == target_class) / pred_class.size
    assert (predictions.size == targets.size), "ERROR! prediction and target size mismatch"

    accr = np.sum(predictions == targets) / predictions.size

    return accr


def evaluate(model, criterion, eval_data):
    test_loss = []
    accr = []

    # Load the test data
    data_loader = DataLoader(eval_data, collate_fn=TweetsBaseDataset.collate_fn,
                             batch_size=TEST_BATCH_SIZE, shuffle=True)

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(device)

            pred = model.forward(inputs)
            accr.append(accuracy(pred.cpu().numpy(), labels))

            labels = torch.LongTensor(labels).to(device)
            loss = criterion(pred, labels)
            test_loss.append(loss.item())

    return np.average(test_loss), np.average(accr)

def train_test(train_data, eval_data, test_data):
    # Set the seed for reproduction of results
    torch.manual_seed(123)

    # Load the NN model
    model = Dummy_network().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load the data using the data loader module
    data_loader = DataLoader(train_data, collate_fn=TweetsBaseDataset.collate_fn,
                             batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(NUM_EPOCHS):
        iteration = 1
        for inputs, labels, _ in data_loader:
            # TODO: If one is using torch models, the data might require some reshaping
            inputs = inputs.to(device)
            labels = torch.LongTensor(labels).to(device)

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Run the model
            out = model(inputs)

            # Compute the losses
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            accr = accuracy(out.cpu().numpy(), labels.cpu().numpy())
            train_accuracy.append(accr)
            print("\rEpoch {}/{}: Training loss = {}, Training accuracy = {}".format(epoch,
                        iteration, "%.4f" % loss, "%.4f" % accr), end='', flush=True)

            iteration += 1

        # Do the evaluation
        eval_loss, eval_accr = evaluate(model, criterion, eval_data)
        val_loss.append(eval_loss)
        val_accuracy.append(eval_accr)
        print("\nValidation loss = {}, Validation accuracy = {}".format("%.4f" % eval_loss, "%.4f" % eval_accr))

    print("Training Completed")

    # Test the trained model
    test_loss, test_accuracy = evaluate(model, criterion, test_data)
    print("\nTest loss = {}, Test accuracy = {}".format("%.4f" % test_loss, "%.4f" % test_accuracy))
    print("Test Completed")


def main():
    if not os.path.exists(DATA_DIR_DEFAULT):
        os.makedirs(DATA_DIR_DEFAULT)

    english_train = TweetsBOWDataset.load(os.path.join(DATA_DIR_DEFAULT, "us_train.set"))
    english_dev = TweetsBaseDataset.load(os.path.join(DATA_DIR_DEFAULT, "us_dev.set"))
    english_test = TweetsBOWDataset.load(os.path.join(DATA_DIR_DEFAULT, "us_test.set"))

    train_test(english_train, english_dev, english_test)

if __name__ == "__main__":
    # TODO: May be take the arguments for hyperparameters
    main()