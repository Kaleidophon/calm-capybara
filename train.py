"""
Training and evaluation functions for the emoji prediction task
"""
import os

# Torch modules
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

# Data loading modules
from tweet_data import TweetsBOWDataset, TweetsBaseDataset

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from tensorboardX import SummaryWriter


# Directory in which tweet data is saved
DATA_DIR_DEFAULT = './data'

TEST_BATCH_SIZE = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_score(logits, targets):
    """
    Computes the score of the network given output logits.
    Args:
        - logits (tensor): predictions of the model.
            shape (n_batches, n_classes)
        - targets (tensor): true labels, containing values in [0, n_classes-1]
            shape (n_batches,)
        - score (str): one of 'accuracy', 'f1_score'
    Returns: float, the calculated score.
    """
    targets = targets.data.cpu().numpy()
    predictions = torch.argmax(logits, dim=1).data.cpu().numpy()

    return f1_score(targets, predictions, average='macro')


def evaluate(model, criterion, eval_data, score='f1_score'):
    """
    Calculate a classification score of the model on a given dataset.
    Args:
        model (torch.nn.Module): the model to evaluate
        criterion (torch.nn.CrossEntropyLoss): used to calculate the loss
        eval_data (TweetsBaseDataset): dataset on which to evaluate the model
        score (str): one of 'f1_score', 'precision', 'recall'
    Returns:
        mean_loss (float), mean loss on the dataset
        score (float)
    """
    if score == 'f1_score':
        score_fn = f1_score
    elif score == 'precision':
        score_fn = precision_score
    elif score == 'recall':
        score_fn = recall_score
    else:
        raise ValueError('Invalid score: {}'.format(score))

    model.eval()
    mean_loss = 0

    # Load the test data
    data_loader = DataLoader(eval_data, collate_fn=TweetsBaseDataset.collate_fn,
                             batch_size=TEST_BATCH_SIZE, shuffle=True)

    y_true = np.empty(len(eval_data), dtype=int)
    y_pred = np.empty(len(eval_data), dtype=int)
    counter = 0

    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(inputs, lengths)

            loss = criterion(outputs, labels)
            mean_loss += loss.item()/len(data_loader)

            predictions = torch.argmax(outputs, dim=1).data.cpu().numpy()
            y_pred[counter:counter + len(labels)] = predictions
            y_true[counter:counter + len(labels)] = labels.data.cpu().numpy()

            counter += len(labels)

    score = score_fn(y_true, y_pred, average='macro')

    return mean_loss, score


def train_model(model, datasets, batch_size, epochs, learning_rate,
                weight_decay=0, metadata=None, weights=None, checkpoint=None):
    """Train a sequence model on the Emoji Dataset.
    Args:
        model (torch.nn.Module): the model to be trained
        datasets (tuple): contains 3 datasets (TweetsBaseDataset)
            corresponding to train, dev and test splits
        batch_size (int): mini-batch size for training
        epochs (int): number of iterations over the training set
        learning_rate (float): used in the optimizer
        weight_decay (float): regularization factor for the optimizer
        metadata (dict): contains keys and values of any type with a valid
            string representation, which are saved for visualization in
            TensorBoard. Use to log model name and hyperparameters
        weights (dict): maps strings to weights (torch.tensor) to be
            visualized as histograms in TensorBoard
        checkpoint (str): path of an existing checkpoint (.pt) file
    Returns:
        tuple, containing best validation F1 score and test F1 score
    """
    train_set, dev_set, test_set = datasets
    train_loader = DataLoader(train_set, batch_size, shuffle=True,
                              num_workers=4,
                              collate_fn=TweetsBaseDataset.collate_fn)

    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    if checkpoint is not None:
        load_model(model, optimizer, checkpoint, eval_model=False)

    # A writer to save TensorBoard events
    writer = SummaryWriter()
    logdir = writer.file_writer.get_logdir()

    # Write hyperparameters to summary
    if metadata is None:
        metadata = {}
    metadata['Batch size'] = batch_size
    metadata['Learning rate'] = learning_rate
    text_summary = _build_text_summary(metadata)
    writer.add_text('metadata', text_summary)

    best_score = 0
    test_f1 = 0
    best_ckpt_link = os.path.join(logdir, 'best-ckpt.pt')

    try:
        steps = 0
        for epoch in range(1, epochs + 1):
            model.train()
            print('Epoch {:d}/{:d}'.format(epoch, epochs))
            n_batches = 0
            for inputs, labels, lengths in train_loader:
                steps += 1
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

                # Log scores on training set
                if n_batches % 100 == 0:
                    f1 = _get_score(outputs, labels)
                    print("\r{}/{}: loss = {:.4f}, f1_score = {:.4f}".format(
                        n_batches, len(train_loader), loss, f1),
                        end='', flush=True)

                    # Write metrics to TensorBoard
                    writer.add_scalar('training/loss', loss, steps)
                    writer.add_scalar('training/f1_score', f1, steps)
                    # Write histograms
                    if weights is not None:
                        for name, data in weights.items():
                            writer.add_histogram('weights/' + name, data, steps)

            # Evaluate on dev set
            eval_loss, eval_f1 = evaluate(model, criterion, dev_set)
            print("\nvalidation loss = {:.4f}, validation f1_score = {:.4f}".format(
                eval_loss, eval_f1))

            # Write to Tensorboard
            writer.add_scalar('validation/loss', eval_loss, steps)
            writer.add_scalar('validation/f1_score', eval_f1, steps)

            # Save the checkpoint
            ckpt_path = os.path.join(logdir, 'ckpt-{:d}.pt'.format(epoch))
            save_model(model, optimizer, epoch, ckpt_path)

            # Create a symbolic link to the best model
            if eval_f1 > best_score:
                best_score = eval_f1
                if os.path.islink(best_ckpt_link):
                    os.unlink(best_ckpt_link)
                os.symlink(os.path.basename(ckpt_path), best_ckpt_link)

        print("Training Completed. Evaluating on test set...")

        # Evaluate on test set
        test_loss, test_f1 = evaluate(model, criterion, test_set)
        print("\ntest loss = {:.4f}, test f1_score = {:.4f}".format(
            test_loss, test_f1))

        # Write to Tensorboard
        writer.add_scalar('test/loss', test_loss, 0)
        writer.add_scalar('test/f1_score', test_f1, 0)

    except KeyboardInterrupt:
        print('Interrupted training.')

    return best_score, test_f1


def save_model(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch}, path)


def load_model(model, optimizer, checkpoint, eval_model=True):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    if eval_model:
        model.eval()
    else:
        model.train()

    return model, optimizer, epoch


def _build_text_summary(metadata):
    text_summary = ""
    for key, value in metadata.items():
        text_summary += '**' + str(key) + ':** ' + str(value) + '</br>'
    return text_summary
