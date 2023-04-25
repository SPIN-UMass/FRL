# Federated Rank Learning (FRL) 

This repository contains the implementation of Federated Rank Learning (FRL) algorithm, as described in our paper: Every Vote Counts: Ranking-Based Training of Federated Learning to Resist Poisoning Attacks, to appear on USENIX Security Symposium 2023. 

## Getting Started

### Prerequisites

Please make sure you have the following dependencies installed:

- Python 3.10.9
- Torch 1.13.1 
- Numpy 1.23.5
- Torchvision 0.14.1

### Installation

- I) To download the repository, use the following command:

```bash
git clone https://github.com/SPIN-UMass/FRL.git
```

- II) Create a new conda environment. you can do so using the following command:

```bash
conda create --name FRL_test python=3.10.9
```

- III) Activate the environment:

```bash
conda activate FRL_test
```

- IV) Then, to install the dependencies, run:

```bash
pip install -r requirements.txt
```



## Basic Test

To run a simple experiment on CIFAR10, please run the following command:


```bash
python main.py --data_loc "/CIFAR10/data/" --config experiments/001_config_CIFAR10_Conv8_FRL_1000users_noniid1.0_nomalicious.txt
```

- Note that argument 'data_loc' shows the path to dataset storage (for creation or existing dataset).

This will distribute CIFAR10 over 1000 clients in a non-iid fashion with a Dirichlet distribution parameter $\beta=1.0$. Then, a federated rank learning will be run on top of these 1000 users for 2000 global FL rounds, where 25 clients are chosen for their local update in each round.

## Citation

```
@inproceedings{mozaffarievery,
  title={Every Vote Counts: Ranking-Based Training of Federated Learning to Resist Poisoning Attacks},
  author={Mozaffari, Hamid and Shejwalkar, Virat and Houmansadr, Amir},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  year={2023}
}
```
