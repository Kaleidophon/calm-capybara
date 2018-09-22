"""
Training and evaluation functions for the emoji prediction task
"""
# Torch modules
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Data loading modules
from tweet_data import TweetsBOWDataset, TweetsBaseDataset

from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter


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
        return f1_score(targets.data.numpy(), predictions.data.numpy(),
                        average='macro')

def evaluate(model, criterion, eval_data):
    # TODO: replace with continuous averaging
    mean_loss = 0
    mean_accr = 0

    # Load the test data
    data_loader = DataLoader(eval_data, collate_fn=TweetsBaseDataset.collate_fn,
                             batch_size=TEST_BATCH_SIZE, shuffle=True)

    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            mean_accr += get_score(outputs, labels, 'accuracy')/len(data_loader)

            loss = criterion(outputs, labels)
            mean_loss += loss.item()/len(data_loader)

    return mean_loss, mean_accr

def train_model(model, datasets, batch_size, epochs, learning_rate,
                hparams=None):
    train_set, dev_set, test_set = datasets

    train_loader = DataLoader(train_set, batch_size, shuffle=True,
                              num_workers=4,
                              collate_fn=TweetsBaseDataset.collate_fn)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter()

    for epoch in range(epochs):
        print('Epoch {:d}/{:d}'.format(epoch, epochs))
        n_batches = 0
        for inputs, labels, lengths in train_loader:
            n_batches += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Run the model
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)

            # Optimize
            loss.backward()
            optimizer.step()

            # Evaluate on training set
            if n_batches % 10 == 0:
                writer.add_scalar('loss', loss, n_batches)
                score = get_score(outputs, labels, score='accuracy')
                print("\r{}/{}: loss = {:.4f}, accuracy = {:.4f}".format(
                    n_batches, len(train_loader), loss, score),
                    end='', flush=True)

        # Evaluate on dev set
        # TODO: replace/add f1_score
        eval_loss, eval_accr = evaluate(model, criterion, dev_set)
        print("\nValidation loss = {:.4f}, Validation accuracy = {:.4f}".format(
            eval_loss, eval_accr))

    print("Training Completed")

    # Evaluate on test set
    # TODO: replace/add f1_score
    test_loss, test_accr = evaluate(model, criterion, dev_set)
    print("\nTest loss = {:.4f}, Test accuracy = {:.4f}".format(
        test_loss, test_accr))
    print("Test Completed")
