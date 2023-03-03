# Federated Rank Learning (FRL) 

This repository contains the implementation of Federated Rank Learning (FRL) algorithm, as described in our paper: Every Vote Counts: Ranking-Based Training of Federated Learning to Resist Poisoning Attacks, to appear on USENIX Security Symposium 2023. 

## Getting Started

### Prerequisites

Please make sure you have the following dependencies installed:

- PyTorch 1.13.1 
- Numpy 1.23.5

### Installation

To download the repository, use the following command:

git clone https://github.com/SPIN-UMass/FRL.git

Then, to install the dependencies, run:

pip install -r requirements.txt

## Basic Test

To run a simple experiment on CIFAR10, please run the following command:

python experiments/001_config_CIFAR10_Conv8_FRL_1000users_noniid1.0_nomalicious.txt


This will distribute CIFAR10 over 1000 clients in a non-iid fashion with a Dirichlet distribution parameter $\beta=1.0$. Then, a federated rank learning will be run on top of these 1000 users for 2000 global FL rounds, where 25 clients are chosen for their local update in each round.
