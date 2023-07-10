# import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision
# from jax import numpy as jnp
from sklearn.utils import shuffle
from tqdm import tqdm


class PostNet:

    def __init__(self, sample_list = []):
        if len(sample_list) == 0:
            raise ValueError("sample_list must be non-empty.")
        self.sample_list = sample_list

    def __call__(self, x):
        with torch.no_grad():
            output_avg = 0
            for net in self.sample_list:
                output_avg += torch.exp(F.log_softmax(net(x).detach(), dim=1))  # more stable than softmax
        return output_avg / len(self.sample_list)

    def predict(self, x, k = 1):
        return torch.argsort(self(x), dim=-1, descending=True)[:, :k]


class PostNetJax:

    def __init__(self, net_apply, get_params_and_state, path = None, t_burn_in = 50):
        if path is None:
            raise ValueError("We need the path to the checkpoints.")
        self.path = path
        self.t_burn_in = t_burn_in
        self.get_params_and_state = get_params_and_state
        self.net_apply = net_apply
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # todo: just to avoid a bug

    def __call__(self, x):
        # todo: modify to get faster results
        x = np.array(x.cpu().numpy().transpose(0, 2, 3, 1), dtype=jnp.float32)
        output_avg = None
        count = 0
        for chain_id in range(3):
            for sample_id in range(self.t_burn_in, 291):
                # if sample_id > self.t_burn_in:  # todo: to remove
                #     break
                params, net_state = self.get_params_and_state(self.path, chain_id, sample_id)
                preds, _ = self.net_apply(params, net_state, None, (x, None), False)
                # get log probs
                log_probs = jax.nn.log_softmax(preds, axis=1)
                # get preds
                if output_avg is None:
                    output_avg = jnp.exp(log_probs)
                else:
                    output_avg += jnp.exp(log_probs)
                count += 1
        return torch.from_numpy(np.asarray(output_avg)).to(self.device) / count  # todo: bad but...


def confidence(ytest, all_probs, tau_list = np.linspace(0, 1, num=100)):
    # Compute the accuracy in function of p(y|x)>tau
    accuracies = np.ones_like(tau_list)
    ind_pred = np.argmax(all_probs, axis=1)
    prob_preds = np.max(all_probs, axis=1)
    label_pred = np.take_along_axis(np.arange(all_probs.shape[1]), ind_pred, axis=0)
    # ytest = testloader.dataset.targets
    misclassified = np.where(label_pred != ytest)[0]
    for (i, tau) in enumerate(tau_list):
        ind_tau = np.where(prob_preds >= tau)[0]
        if len(ind_tau) > 0:
            ind_inter = np.intersect1d(ind_tau, misclassified)
            accuracies[i] = 1 - len(ind_inter) / len(ind_tau)
    return accuracies, misclassified


def ECE(all_probs, ytest, num_bins = 1, print_ = True):
    # Compute the Expected Calibration Error (ECE)
    prob_preds = np.max(all_probs, 1)
    predictions = np.argmax(all_probs, 1)
    accuracies = (predictions == ytest)
    ece = 0.
    for it in range(num_bins):
        ind_bin = (it / num_bins < prob_preds) * (prob_preds <= (it + 1) / num_bins)
        if not ind_bin.any():
            continue
        acc_bin = accuracies[ind_bin]
        prob_bin = prob_preds[ind_bin]
        ece_bin = ind_bin.sum() * np.abs(acc_bin.mean() - prob_bin.mean())
        ece += ece_bin
    if print_:
        print("ECE =", ece / len(ytest))
    return ece / len(ytest)


def BS(ytest, all_probs, print_ = True):
    # ytest = testloader.dataset.targets
    num_classes = len(np.unique(ytest))
    # Perform a one-hot encoding
    labels_true = np.eye(num_classes)[ytest]
    # Compute the Brier Score (BS)
    bs = num_classes * np.mean((all_probs - labels_true) ** 2)
    if print_:
        print("BS =", bs)
    return bs


def NLL(all_probs, ytest, print_ = True):
    # Compute the Negative Log Likelihood (NLL)
    log_it = - np.log(np.take_along_axis(all_probs, np.expand_dims(ytest, axis=1), axis=1)).squeeze()
    nll = log_it.mean()
    if print_:
        print(f"NNL = {nll}")
    return nll


def Predictive_entropy(ytest, all_probs, post_net, transform, dataset_name = "CIFAR10", path_dataset_ood = None,
                       path_fig = None):
    # Compute the Negative Log Likelihood (NLL)
    # ytest = testloader.dataset.targets
    log_it = - np.log(np.take_along_axis(all_probs, np.expand_dims(ytest, axis=1), axis=1)).squeeze()
    nll = log_it.mean()
    entropy_dataset = - (all_probs * np.log(all_probs)).sum(axis=1)

    # Create the OOD dataset
    dataset_ood_name = "SVHN" if dataset_name != "MNIST" else "FashionMNIST"
    dataset_ood = getattr(torchvision.datasets, dataset_ood_name)
    testset_ood = dataset_ood(root=path_dataset_ood, transform=transform, download=True)
    ood_test = testset_ood.labels if dataset_name != "MNIST" else testset_ood.targets

    # Define the OOD test loader
    loader_test_ood = torch.utils.data.DataLoader(testset_ood, batch_size=500, shuffle=False)
    # Compute the predicted probabilities
    predicted_ood = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for images, labels in tqdm(loader_test_ood):
            images, labels = images.to(device), labels.to(device)
            outputs = post_net(images).cpu().numpy()
            if predicted_ood is None:
                predicted_ood = outputs
            else:
                predicted_ood = np.vstack((predicted_ood, outputs))

    # Compute the Negative Log Likelihood (NLL) on MNIST
    # entropy_ood = - np.log(np.take_along_axis(predicted_ood, np.expand_dims(ood_test, axis=1), axis=1)).squeeze()
    entropy_ood = - (predicted_ood * np.log(predicted_ood)).sum(axis=1)

    # Store the results in a dictionary    
    entropy_dict = {"Dataset": entropy_dataset, "OOD dataset": entropy_ood}

    # Display the predicted entropies
    if path_fig is not None:
        ax = sns.kdeplot(data=entropy_dict, fill=True, cut=0, common_norm=False)
        ax.set(xlabel='pred. entropy', ylabel='Density')  # title='')
        plt.savefig(path_fig, bbox_inches='tight')

    return entropy_dict, nll


def AUC(entropy_dataset, entropy_ood, print_ = True):
    # Compute the Area Under the Curve (AUC), ie prob(entropy_dataset <= entropy_ood)
    auc = 0
    for y in entropy_dataset:
        auc += np.sum(y <= entropy_ood)
    auc /= len(entropy_dataset) * len(entropy_ood)
    if print_:
        print(f"AUC = {auc}")
    return auc


def accuracy_confidence(all_probs, ytest, tau_list, num_bins = 1):
    score = np.zeros_like(tau_list)
    all_probs, ytest = shuffle(all_probs, ytest)
    all_probs_split, ytest_split = np.array_split(all_probs, num_bins), np.array_split(ytest, num_bins)
    for (probs, y) in zip(all_probs_split, ytest_split):
        acc, miss = confidence(y, probs, tau_list)
        prob_preds = np.max(probs, axis=1)
        score += np.array([acc[it] - prob_preds[prob_preds >= tau].mean() for it, tau in enumerate(tau_list)])
    return score / num_bins


def calibration_curve(all_probs, ytest, num_bins = 1):
    prob_preds = np.max(all_probs, 1)
    predictions = np.argmax(all_probs, 1)
    ind_order = np.argsort(prob_preds)
    ind_bins = np.array_split(ind_order, num_bins)
    bins = np.array_split(prob_preds[ind_order], num_bins)
    accuracies = (predictions == ytest)
    accuracy_bin_array = np.zeros(num_bins)
    confidence_bin_array = np.zeros(num_bins)
    for it, bin_prob in enumerate(bins):
        accuracy_bin = accuracies[ind_bins[it]].mean()
        confidence_bin = bin_prob.mean()
        accuracy_bin_array[it] = accuracy_bin
        confidence_bin_array[it] = confidence_bin
    return accuracy_bin_array, confidence_bin_array
