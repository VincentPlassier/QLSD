import numpy as np
import copy
import torch
import torch.nn.functional as F


def agreement(predictions: np.array, reference: np.array):
    """Returns 1 if predictions match and 0 otherwise."""
    return (predictions.argmax(axis=-1) == reference.argmax(axis=-1)).mean()


def total_variation_distance(predictions: np.array, reference: np.array):
    """Returns total variation distance."""
    return np.abs(predictions - reference).sum(axis=-1).mean() / 2.


def get_accuracy_fn(net_fn, batch):
    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    # get logits
    with torch.no_grad():
        logits = net_fn(x)
        # get log probs 
        log_probs = F.log_softmax(logits, dim=1)
        # get preds 
        probs = torch.exp(log_probs)
        preds = torch.argmax(logits, dim=1)
    correct = (predicted == labels).sum().item()
    return correct, probs


def evaluate_fn(net_fn, data_loader):
    correct = 0
    total = 0
    all_probs = []
    net_fn.eval()
    for x, y in data_loader:
        correct_batch, probs_batch = get_accuracy_fn(net_fn, (x, y))
        correct += correct_batch
        total += x.size(0)
        all_probs.append(probs_batch)
    net_fn.train()
    all_probs = torch.cat(all_probs, dim=0)
    return scorrect / total, all_probs
