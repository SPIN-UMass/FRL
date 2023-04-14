import os
from args import args
import random
import numpy as np
import pathlib
import torch

import data
from FL_train import *



def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        
    # Make the a directory corresponding to this run for saving results, checkpoints etc.
    i = 0
    while True:
        run_base_dir = pathlib.Path(f"{args.log_dir}/FRL~try={str(i)}")

        if not run_base_dir.exists():
            os.makedirs(run_base_dir)
            args.name = args.name + f"~try={i}"
            break
        i += 1

    (run_base_dir / "output.txt").write_text(str(args))
    args.run_base_dir = run_base_dir

    print(f"=> Saving data in {run_base_dir}")
    
    
    
    #distribute the dataset
    print ("dataset to use is: ", args.set)
    print ("number of FL clients: ", args.nClients)
    print ("non-iid degree data distribution: ", args.non_iid_degree)
    print ("batch size is : ", args.batch_size)
    print ("test batch size is: ", args.test_batch_size)
    
    data_distributer = getattr(data, args.set)()
    tr_loaders = data_distributer.get_tr_loaders()
    te_loader = data_distributer.get_te_loader()
    
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print ("use_cuda: ", use_cuda)
    
    
    #Federated Learning
    print ("type of conv layer: ", args.FL_type)
    if args.FL_type == "FRL":
        FRL_train(tr_loaders, te_loader)
    elif args.FL_type == "FedAVG":
        FedAVG(tr_loaders, te_loader)
    else:
        FedAVG(tr_loaders, te_loader)

   
if __name__ == "__main__":
    main()