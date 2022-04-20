import time
import torch
from util import AverageMeter
import os 
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def tripletFit(loader, model, loss_fn, optimizer, metrics,epoch,args):
    """
    Train the model with online triplet loss
    Args:
            dataloader (TrainImageFolder): Data loader
            model (nn.Module): Texture Architecture
            loss_fn: OnlineTripletLoss 
            optimizer (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model 
            metrics: AverageNonzeroTripletsMetric
            epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    for metric in metrics:
        metric.reset()
    
    total_acc_pos,total_acc_triplets,total_triplets = 0.0,0.0,0.0
    # switch to train mode
    model.train()
    end = time.time()

    for batch_idx, (data, target,true_lbl) in enumerate(loader):

        data_time.update(time.time() - end)
        
        #Save Checkpoint
        n = len(loader) * epoch + batch_idx
       
        if n % args.checkpoints == 0:
            path = os.path.join(
                args.exp,
                'checkpoints',
                'checkpoint_' + str(n / args.checkpoints) + '.pth.tar',
            )
            if args.verbose:
                print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }, path)


        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (torch.autograd.Variable(data),)
        
        data = tuple(d.cuda() for d in data)
        if target is not None:
            target = torch.autograd.Variable(target)
       
        #Clear the gradient
        optimizer.zero_grad()
      
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target, true_lbl)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        acc_pos, acc_trip, total_triplet = loss_outputs[1] 
        total_acc_pos += acc_pos
        total_acc_triplets += acc_trip
        total_triplets += total_triplet
            
        # record loss
        losses.update(loss.item(), data[0].size(0))
        
        # compute gradient and do SGD step
        loss.backward()   
        optimizer.step() 
       
        for metric in metrics:
            metric(outputs, target, loss_outputs)

        del loss
        del loss_outputs
        del outputs
        del loss_inputs
       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.verbose and (batch_idx % 100) == 0:
            print('Epoch: [{0}][{1}/{2}({3}%)]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, batch_idx * len(data[0]), len(loader.dataset),
                  round((100. * batch_idx / len(loader)),0)
                  , batch_time=batch_time, data_time=data_time, loss=losses))
            
            print('GOOD POSITIVE PAIRS: {:.4f}\t'.format(acc_pos/total_triplet))
            print('GOOD Triplets: {:.4f}\t'.format(acc_trip/total_triplet))
        
    if args.verbose:
        print('Epoch: [{0}][{1}/{2}({3}%)]\t'
	  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
	  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
	  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
	  .format(epoch, batch_idx * len(data[0]), len(loader.dataset),
	  round((100. * batch_idx / len(loader)),0)
	  , batch_time=batch_time, data_time=data_time, loss=losses))
         
        print('AVG GOOD POSITIVE PAIRS: {:.4f}\t'.format(total_acc_pos/total_triplets))
        print('AVG GOOD Triplets: {:.4f}\t'.format(total_acc_triplets/total_triplets))   

    return losses.avg

