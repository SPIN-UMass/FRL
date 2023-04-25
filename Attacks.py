from args import args
from AGRs import *
from eval import *
from misc import *
import torch
import pickle
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np

def our_attack_trmean(all_updates, n_attackers, dev_type='sign', threshold=5.0, threshold_diff=1e-5):
    
    model_re = torch.mean(all_updates, 0)
    
    if dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([threshold]).cuda()  # compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = threshold_diff
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = tr_mean(mal_updates, n_attackers)

        loss = torch.norm(agg_grads - model_re)

        if prev_loss < loss:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - lamda_succ * deviation)
    
    return mal_update



def our_attack_mkrum(all_updates, model_re, n_attackers,dev_type='unit_vec', threshold=5.0, threshold_diff=1e-5):

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)
        
    lamda = torch.Tensor([threshold]).cuda() #compute_lambda_our(all_updates, model_re, n_attackers)
    # print(lamda)
    # threshold_diff = 1e-7
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads, krum_candidate = multi_krum(mal_updates, n_attackers, multi_k=True)

        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    # print('lambda succ ', lamda_succ)
    mal_update = (model_re - lamda_succ * deviation)
    return mal_update
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates