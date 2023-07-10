import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import shuffle
from termcolor import cprint
from torch.utils.data import DataLoader, TensorDataset

from bayesian_benchmarks.data import (_ALL_CLASSIFICATION_DATATSETS, _ALL_REGRESSION_DATATSETS, get_classification_data,
                                      get_regression_data)


def pprint(text):
    cprint(text, 'red', attrs=['bold', 'underline'])
    # print(colored(text, 'red', attrs=['bold', 'underline']))


## Define a fonction to display the contour of Gaussian densities
def gauss_draw(mu, sigma, axis, levels = 1, colors = 'grey', color_mu = 'cyan', linestyles = 'solid'):
    # compute a space grid
    Zx, Zy = np.meshgrid(np.linspace(axis[0], axis[1], num=100), np.linspace(axis[2], axis[3], num=100))
    Z = np.stack((Zx, Zy)).T.reshape(-1, 2)

    # define the considered pdf
    def pdf(x): return ss.multivariate_normal.pdf(x, mean=mu, cov=sigma)

    # Y contains the values of the density calculated
    Y = np.fromiter(map(pdf, Z), dtype='float').reshape(Zx.shape).T
    # fig = plt.figure()
    fig = None
    # plot the contours
    # CS = plt.contour(Zx, Zy, Y, levels, colors=colors, linestyles=linestyles)
    # plt.clabel(CS, inline=1, fontsize=4)
    # display the center
    plt.plot(mu[0], mu[1], '*', color=color_mu, markersize=6)
    return fig


def plot_result(param_list, axis, mu, pca = None):
    n_components = 2
    param_list = PCA(n_components).fit_transform(param_list)
    # if pca is not None:
    # param_list = pca.transform(param_list)
    # mu = pca.transform(mu.reshape(1, -1)).flatten()
    ax = plt.subplot()
    # Display the results
    ax.hist2d(param_list[:, 0], param_list[:, 1], bins=30, cmap='Blues', density=True)
    ax.axis(axis)
    # plt.colorbar()
    # gauss_draw(mu, np.eye(2), axis, levels=5, colors='red', color_mu='khaki')
    # plt.colorbar()
    plt.show()


def plot_samples(param_list, n_components = 2, axis = None):
    param_list = PCA(n_components).fit_transform(param_list)
    # Display the results
    ax = plt.subplot()
    ax.hist2d(param_list[:, 0], param_list[:, 1], bins=30, cmap='Blues', density=True)
    if axis is not None:
        ax.axis(axis)
    plt.show()


def rotation(phi):
    c, s = np.cos(phi), np.sin(phi)
    mat = c * np.eye(2)
    mat[0, 1] = s
    mat[1, 0] = - s
    return mat


def create_cov(dim, scale, psi = 2 * np.pi, rng = None):
    mat = np.eye(dim)
    for i in range(dim // 2):
        r = rotation(rng.uniform(psi))
        mat[2 * i: 2 * (i + 1), 2 * i: 2 * (i + 1)] = r.dot(np.diag(rng.uniform(size=2)).dot(r.T))
    return scale * mat


def generate_heterogeneous_classification(inputs, targets, num_clients, transform = None,
                                          rng = np.random.default_rng(0)):
    datasets_dict = {}
    for label in np.unique(targets):
        idx_label = np.where(targets == label)[0]
        p = rng.uniform(0, 1, size=num_clients)
        num_data = (1 + np.round((len(idx_label) - 1 * num_clients) * rng.dirichlet(p))).astype('int')
        num_total = 0
        for client, num in enumerate(num_data):
            if client == num_clients - 1:
                num = len(idx_label) - num_total
            idx_client = idx_label[num_total: num_total + num]
            if num == 0:
                continue
            if client not in datasets_dict.keys():
                datasets_dict[client] = [inputs[idx_client], targets[idx_client]]
            else:
                datasets_dict[client][0] = np.vstack((datasets_dict[client][0], inputs[idx_client]))
                datasets_dict[client][1] = np.hstack((datasets_dict[client][1], targets[idx_client]))
            num_total += num
    datasets = []
    for x, y in datasets_dict.values():
        if transform == torch.from_numpy:
            datasets.append([torch.from_numpy(x), torch.from_numpy(y)])
        elif transform is not None:
            datasets.append([x, transform(y.reshape(-1, 1))])
        else:
            datasets.append([x, y])
    return datasets


def generate_heterogeneous_regression(inputs, targets, num_clients, transform = torch.from_numpy,
                                      rng = np.random.default_rng(0)):
    p = rng.uniform(.5, .5, size=num_clients)  # to add heterogeneity
    num_data = np.cumsum(len(targets) * rng.dirichlet(p)).astype('int')
    num_data = np.hstack(([0], num_data))
    num_data[-1] = len(targets)

    datasets = []
    for i in range(num_clients):
        inpts = transform(inputs[num_data[i]: num_data[i + 1]])
        targts = transform(targets[num_data[i]: num_data[i + 1]])
        datasets.append([inpts, targts])

    return datasets


def load_UCI_dataset(dataset_name, rng = None, prop = .9, random_state = 0):
    print('Preparing dataset %s' % dataset_name)

    if dataset_name in _ALL_CLASSIFICATION_DATATSETS:
        dataset = get_classification_data(dataset_name, rng, prop)
    elif dataset_name in _ALL_REGRESSION_DATATSETS:
        dataset = get_regression_data(dataset_name, rng, prop)
    else:
        raise NameError('Invalid dataset_name.')

    print(f'Statistics: N={dataset.N}, D={dataset.D}, Xtrain={dataset.X_train.shape}')

    if dataset_name in _ALL_CLASSIFICATION_DATATSETS:
        Xtrain = dataset.X_train
        Ytrain = dataset.Y_train.ravel()
        le = LabelEncoder()
        le.fit(Ytrain)
        Ytrain = le.transform(Ytrain)
        Ytest = le.transform(dataset.Y_test.ravel())

    elif dataset_name in _ALL_REGRESSION_DATATSETS:
        Xtrain, Ytrain = shuffle(dataset.X_train, dataset.Y_train, random_state=random_state)
        scalerY = StandardScaler()
        scalerY.fit(np.concatenate((dataset.Y_train, dataset.Y_test)))
        Ytrain = scalerY.transform(Ytrain)
        Ytest = scalerY.transform(dataset.Y_test)

    scalerX = StandardScaler()
    scalerX.fit(np.concatenate((Xtrain, dataset.X_test)))
    Xtrain = scalerX.transform(Xtrain)
    Xtest = scalerX.transform(dataset.X_test) if len(dataset.Y_test) > 0 else dataset.X_test

    return Xtrain, Ytrain, Xtest, Ytest


# Define the title to save the results
def create_title(mc_iter, num_clients, factor, pc, tau, exp_num = None):
    title = str()
    for key, value in {'mc_':     mc_iter, 'c_': num_clients, 'f_': factor, 'pc_': pc, 'tau_': tau,
                       'expNum_': exp_num}.items():
        if key == 'expNum_' and value is None:
            continue
        title += key + str(value) + '-'
    return title[:-1]


def plot_save_fig(mse_dict, mc_iter, burn_in, title, xlog = True):
    plt.clf()
    if xlog:
        plt.xscale('log')
    plt.yscale('log')
    plt.grid('True')
    for key, mse in mse_dict.items():
        plt.plot(np.arange(burn_in, mc_iter + 1) + 1, mse.mean(axis=0), label=key)
    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('MSE comparison')
    plt.legend()
    plt.savefig(title + '-mse_fig.pdf', bbox_inches='tight')
    plt.clf()


def fusion(loader0, loader1, tau = 0):
    length0, length1 = int((1 - tau) * len(loader0.dataset)), int(tau * len(loader1.dataset))
    X0, Y0 = loader0.dataset.data, loader0.dataset.targets
    X1, Y1 = loader1.dataset.data, loader1.dataset.targets  # TODO: verify for CIFAR10
    # if length0 == 0:
    #     return loader1
    # elif length1 == 0:
    #     return loader0
    idx0 = np.random.choice(len(Y0), size=length0, replace=False)
    idx1 = np.random.choice(len(Y1), size=length1, replace=False)
    transform = loader0.dataset.transform
    X = torch.cat((X0[idx0], X1[idx1])).data.numpy()
    X = transform(X.transpose(1, 2, 0))[:,np.newaxis]
    Y = torch.cat((Y0[idx0], Y1[idx1]))
    return DataLoader(TensorDataset(X, Y), loader0.batch_size, shuffle=False)
