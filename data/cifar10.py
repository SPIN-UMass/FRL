from args import args
from torchvision import datasets, transforms
import torchvision
from data.Dirichlet_noniid import *

class CIFAR10:
    def __init__(self):
        
        args.output_size = 10
        
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_loc,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_loc,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )

        tr_per_participant_list, tr_diversity = sample_dirichlet_train_data_train(train_dataset, args.nClients, alpha=args.non_iid_degree, force=False)

        self.tr_loaders = []
        tr_count = 0
        for pos, indices in tr_per_participant_list.items():
            if len(indices)==1 or len(indices)==0:
                print (pos)
            tr_count += len(indices)
            batch_size = args.batch_size
            self.tr_loaders.append(get_train(train_dataset, indices, args.batch_size))
#         print ("number of total training points:" ,tr_count)
        self.te_loader= torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    

    def get_tr_loaders(self):
        return self.tr_loaders
    
    def get_te_loader(self):
        return self.te_loader