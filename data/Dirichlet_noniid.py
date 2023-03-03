from args import args
import pickle
from collections import defaultdict
import os
import random
import numpy as np
import torch

def get_train(dataset, indices, batch_size=args.batch_size, shuffle=True):
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    
    return train_loader

def sample_dirichlet_train_data_train(train_dataset, no_participants, alpha=args.non_iid_degree, force=False):
    file_add = '%s_train_dirichlet_a_%.1f_n%d.pkl'%(args.set, alpha, no_participants)
    
    if not os.path.exists(file_add) or force:
        print('generating participant indices for alpha %.1f' % alpha)

        tr_classes = {}

        for ind, x in enumerate(train_dataset):
            _, label = x
            if label in tr_classes:
                tr_classes[label].append(ind)
            else:
                tr_classes[label] = [ind]

        tr_per_participant_list = defaultdict(list)
        tr_per_participant_list_labels_fr = defaultdict(defaultdict)

        tr_no_classes = len(tr_classes.keys())

        for n in range(tr_no_classes):
            random.shuffle(tr_classes[n])

            tr_class_size=len(tr_classes[n])
            d_sample = np.random.dirichlet(np.array(no_participants * [alpha]))
            tr_sampled_probabilities = tr_class_size * d_sample ##prob of selecting for this class
            for user in range(no_participants):
                no_imgs = int(round(tr_sampled_probabilities[user]))
                sampled_list = tr_classes[n][:min(len(tr_classes[n]), no_imgs)]
                random.shuffle(sampled_list)
                tr_per_participant_list_labels_fr[user][n]=len(sampled_list)
                tr_per_participant_list[user].extend(sampled_list[:])
                tr_classes[n] = tr_classes[n][min(len(tr_classes[n]), no_imgs):]

        with open(file_add, 'wb') as f:
            pickle.dump([tr_per_participant_list, tr_per_participant_list_labels_fr], f)
    else:
        [tr_per_participant_list, tr_per_participant_list_labels_fr] = pickle.load(open(file_add, 'rb'))

    return tr_per_participant_list, tr_per_participant_list_labels_fr