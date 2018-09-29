from tweet_data import TweetsBaseDataset
from torch.utils.data import DataLoader
import numpy as np
import torch

def predict_dataset(model, dataset):
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False,
                             collate_fn=TweetsBaseDataset.collate_fn,
                             num_workers=4)

    y_pred = np.empty(len(dataset), dtype=int)
    y_true = np.empty(len(dataset), dtype=int)
    all_indices = np.empty(len(dataset), dtype=int)
    counter = 0

    with torch.no_grad():
        for inputs, labels, lengths, indices in data_loader:
            outputs = model(inputs, lengths)
            predictions = torch.argmax(outputs, dim=1).data.cpu().numpy()
            y_pred[counter:counter + len(labels)] = predictions
            y_true[counter:counter + len(labels)] = labels.data.cpu().numpy()
            all_indices[counter:counter + len(labels)] = indices

            counter += len(labels)

    return y_true, y_pred, all_indices

