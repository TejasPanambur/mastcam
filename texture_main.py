import argparse
import os
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from TrainImageFolder import TrainImageFolder
from trainer import tripletFit
import clustering
import models
from util import AverageMeter, class_nmi,eval_recall_K,get_acc,plotTSNE


#For Saving Cluster
from saveFunctions import SaveClusters,SaveEpochInfo

#Importing Triplet Libraries
from triplet.datasets import BalancedBatchSampler
from triplet.losses import OnlineTripletLoss, RandomNegativeTripletSelector
from triplet.metrics import AverageNonzeroTripletsMetric

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['texres18'], default='texres18',
                        help='CNN architecture (default: DEP)')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'COP_Kmeans','EliteKmeans'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                        help='number of cluster for k-means (default: 10000)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1.,
                        help="""how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)""")
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--workpath', default='', type=str, metavar='WORKPATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    parser.add_argument('--mode', type=str, choices=['Triplet'])
    parser.add_argument('--margin', type=float,default=0.0)
    parser.add_argument('--niter', type=int, default=1, help='niter (default: 1)')
    return parser.parse_args()    


def objective(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
 
    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    model = torch.nn.DataParallel(model).cuda() 
    #model.to('cuda:0')
    #print(torch.cuda.current_device())
    cudnn.benchmark = True
    
    # create optimizer
    if args.verbose:
        print('Optimizer: SGD')

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=10**args.wd)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        # creating checkpoint repo
        exp_check = os.path.join(args.exp, 'checkpoints')
        if not os.path.isdir(exp_check):
            os.makedirs(exp_check)

    # preprocessing of data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    trans = [transforms.Resize((224,224)),
           transforms.ToTensor(),
           normalize]
   
    # load the data
    end = time.time()

    #dataset = datasets.ImageFolder(args.data, transform=transforms.Compose(tra))
    dataset = TrainImageFolder(args.data, transform=transforms.Compose(trans))   
 
    #print(args.batch)
    if args.verbose:
        print('Load dataset: {0:.2f} s'.format(time.time() - end))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch,
                                             num_workers=args.workers,
                                             pin_memory=True)
    prev_clusters = None
    if args.verbose:
        f = open(args.workpath+"ModelSummary.txt","w")
        f.write(str(model))
        f.close()
    
    # training convnet with DeepCluster
    for epoch in range(args.start_epoch, args.epochs):
    
        end = time.time()
        deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

        if args.verbose:
            print('Length of Dataset:',len(dataset))

        features,lbls,pths = compute_features(dataloader, model, len(dataset),args)
        
        # cluster the features
        clustering_loss = deepcluster.cluster(features,epoch, args,verbose=args.verbose)

        del features

        #assign pseudo-labels
        if args.verbose:
            print('Assign pseudo labels')
        
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, dataset,args)
                                          
        #save Clusterd data 
        SaveClusters(pths,deepcluster.I,epoch,args.workpath)
        if args.verbose:
            print('Saved The Cluster')                                                               
        
        
	    #Batch sampler for triplet training
        if args.mode == 'Triplet':
            train_batch_sampler = BalancedBatchSampler(
                labels = torch.Tensor(np.array([data[1] for data in train_dataset.imgs])),
                n_classes = args.nmb_cluster,
                n_samples = 4
            )
            if args.verbose:
                print('Finished Creating Balanced batches...')

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler = train_batch_sampler,
                num_workers= args.workers,
                pin_memory=True,
            )

        # train network with clusters as pseudo-labels
        end = time.time()


        loss = train(train_dataloader, model, optimizer, epoch, args)

        # print log
        
        if args.verbose:
            print('###### Epoch [{0}] ###### \n'
                  'Time: {1:.3f} s\n'
                  'Clustering loss: {2:.3f} \n'
                  'ConvNet loss: {3:.3f}'
                  .format(epoch, time.time() - end, clustering_loss, loss))
            if epoch!=args.start_epoch:
                nmi = normalized_mutual_info_score(
                    clustering.arrange_clustering(deepcluster.images_lists),
                    prev_clusters
                )
                try:
                    loss_data = loss.data.cpu().numpy()
                except:
                    loss_data = loss
                
                SaveEpochInfo(epoch,clustering_loss,loss_data,nmi,args.workpath)
                if args.verbose:
                    print('NMI against previous assignment: {0:.3f}'.format(nmi))
                del loss_data

            print('####################### \n')
        
        # save running checkpoint
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict()},
                    os.path.join(args.exp, 'checkpoint.pth.tar'))
        
        # save cluster assignments
        prev_clusters = clustering.arrange_clustering(deepcluster.images_lists)
        del loss
        torch.cuda.empty_cache()

    return nmi

def train(loader, model, opt, epoch,args):
    """Training of the model.
        Args:
            loader (TrainImageFolder): Data loader
            model (nn.Module): Texture Architecture
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model 
            epoch (int)
    """
    
    if args.mode == 'Triplet':
        
        loss_fn = OnlineTripletLoss(args.margin,RandomNegativeTripletSelector(args.margin))
        loss = tripletFit(loader=loader,
         model= model, loss_fn= loss_fn,
         metrics = [AverageNonzeroTripletsMetric()],
         optimizer = opt, 
         epoch= epoch, args= args)
		 
    return loss


def compute_features(dataloader, model, N,args):
    """
    Get the feature embedding
    Args:
            dataloader (TrainImageFolder): Data loader
            model (nn.Module): Texture Architecture
            N (int): Number of Images in dataset
    """
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    pths = []
    # discard the label information in the dataloader
    for i, (input_tensor, lbl,pth) in enumerate(dataloader):
        
        with torch.no_grad():
            input_var = input_tensor.cuda()	   
            aux = model(input_var).data.cpu().numpy()
            pths += list(pth)
            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                lbls = np.zeros((N, ), dtype='int32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux
                lbls[i * args.batch: (i + 1) * args.batch] = lbl
            else:
                # special treatment for final batch
                features[i * args.batch:] = aux
                lbls[i * args.batch:] = lbl
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    if args.verbose:
        print('{0} / {1}\t'
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
            .format(i, len(dataloader), batch_time=batch_time))
   
    return features,lbls,pths


def main(args):
   nmi_t = objective(args)
   print('The final nmi at the end of training is:',nmi_t)
       
 
if __name__ == '__main__':
    args = parse_args()
    main(args)


