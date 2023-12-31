# QLSD: Quantized langevin stochastic dynamics for Bayesian federated learning.

This repository contains the code to reproduce the experiments in the paper *QLSD: Quantized langevin stochastic dynamics for Bayesian federated learning* by Vincent Plassier, Maxime Vono, Alain Durmus and Eric Moulines.

## Requirements

We use provide a `requirements.txt` file that can be used to create a conda
environment to run the code in this repo:
```bash
$ conda create --name <env> --file requirements.txt
```

Example set-up using `pip`:
```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Abstract

The objective of Federated Learning (FL) is to perform statistical inference for data which are decentralized and stored locally on networked clients. FL raises many constraints which include privacy and data ownership, communication overhead, statistical heterogeneity, and partial client participation. In this paper, we address these problems in the framework of the Bayesian paradigm. To this end, we propose a novel federated Markov Chain Monte Carlo algorithm, referred to as Quantized Langevin Stochastic Dynamics which may be seen as an extension to the FL setting of Stochastic Gradient Langevin Dynamics, which handles the communication bottleneck using gradient compression. To improve performance, we then introduce variance reduction techniques, which lead to two improved versions coined *QLSD$^{\star}$* and *QLSD$^{++}$. We give both non-asymptotic and asymptotic convergence guarantees for the proposed algorithms.  We illustrate their performances using various Bayesian Federated Learning benchmarks.

## File Structure

```
├── README.md
├── requirements.txt
└── src
    ├── Logistic_regression_EMNIST.ipynb
    ├── models
    │   ├── lenet5.py
    │   ├── logistic.py
    │   ├── mlp.py
    │   ├── pytorch_models.py
    │   └── resnet.py
    ├── logistic_EMNIST.ipynb
    ├── logistic_regression.ipynb
    ├── notebook_logistic_regression.ipynb
    ├── notebook_toy_Gaussian-experiment.ipynb
    ├── qlsdpp.py
    ├── qlsd.py
    ├── qlsd_var.py
    ├── run_qlsd.py
    ├── run_sgd.py
    ├── run_sgd_scores.py
    ├── run_sgld.py
    ├── sgld.py
    ├── ula_fed.py
    └── utils
        ├── compression_algo.py
        ├── learning_rates.py
        ├── load_dataset.py
        ├── metrics.py
        ├── sgld_tools.py
        ├── tools_dl.py
        ├── toy_tools_data.py
        ├── toy_tools_func.py
        └── uncertainties_tools.py
```
