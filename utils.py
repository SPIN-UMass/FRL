from args import args
from eval import *
from misc import *
import torch
import pickle
import torch.nn as nn
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math, copy
import numpy as np


def Find_rank(scores):
    _, idx = scores.detach().flatten().sort()
    return idx.detach()


def FRL_Vote(FLmodel, user_updates, initial_scores):
    for n, m in FLmodel.named_modules():
        if hasattr(m, "scores"):
            args_sorts=torch.sort(user_updates[str(n)])[1]
            sum_args_sorts=torch.sum(args_sorts, 0)
            idxx=torch.sort(sum_args_sorts)[1]
            temp1=m.scores.detach().clone()
            temp1.flatten()[idxx]=initial_scores[str(n)]
            m.scores=torch.nn.Parameter(temp1)                    
            del idxx, temp1
            
            
def train(trainloader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_ind, (inputs, targets) in enumerate(trainloader):

        inputs = inputs.to(device, torch.float)
        targets = targets.to(device, torch.long)

        outputs = model(inputs)
        if len(outputs.shape) == 1:
            outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size()[0])
        top1.update(prec1.item()/100.0, inputs.size()[0])
        top5.update(prec5.item()/100.0, inputs.size()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for batch_ind, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device, torch.float)
            targets = targets.to(device, torch.long)
            outputs = model(inputs)
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(0)
            
            loss = criterion(outputs, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.data, inputs.size()[0])
            top1.update(prec1/100.0, inputs.size()[0])
            top5.update(prec5/100.0, inputs.size()[0])
    return (losses.avg, top1.avg)