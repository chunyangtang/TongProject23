import argparse
import os
import shutil
import colorama
import random
import collections

import numpy as np
import torch

from UR5_Controller import UR5_Controller
from DeepMask import DeepMask
from utils.log_helper import init_log, print_speed, add_file_handler

def robust_set_loss(logProbsNet, gtMasks, step=0.1, iouTh=0.75, maxIter=25,
                    ind=None):
    '''
    logProbsNet: logprobs, i.e, the output of last layer of the network being
                 trained. Should be of the shape [bs,2,ht,wt].
    gtMasks: ground truth mask. Should be of shape [bs,1,ht,wt]. It contains
             either 0 (background) or 1 (foreground).
    step (scalar): granularity of optimization (the lower it is, the finer the
                   results will be).
    iouTh (scalar): lower bound constraint on IoU overlap constraint
    maxIter (scalar): limit on how hardly the constraint should be enforced.
                    Higher means more emphasis on satisfying the constraint.
    output newMasks: new ground truth mask to be trained against.
                    Shape: [bs,1,ht,wt]. 0 (background) or 1 (foreground).
    '''
    if type(step) is np.ndarray:  # tensorflow hacking
        step = step[0]
        iouTh = iouTh[0]
        maxIter = maxIter[0]

    cIn = np.zeros((gtMasks.shape[0],), dtype=np.float)
    cOut = np.zeros((gtMasks.shape[0],), dtype=np.float)
    logProbs = np.copy(logProbsNet)
    indexer = np.zeros(gtMasks.shape, dtype=bool)

    for i in range(maxIter):
        iou_orig = iou(np.argmax(logProbs, axis=-1), gtMasks)
        unconverged = np.logical_not(iou_orig > iouTh)
        if np.all(np.logical_not(unconverged)):
            break

        indexer *= False
        indexer[unconverged] = gtMasks[unconverged] > 0.5
        logProbs[..., 1][indexer] += step
        iou_upIn = iou(np.argmax(logProbs, axis=-1), gtMasks)

        logProbs[..., 1][indexer] -= step
        indexer *= False
        indexer[unconverged] = gtMasks[unconverged] < 0.5
        logProbs[..., 1][indexer] -= step
        iou_downOut = iou(np.argmax(logProbs, axis=-1), gtMasks)

        indexer *= False
        indexer[unconverged] = gtMasks[unconverged] > 0.5
        logProbs[..., 1][indexer] += step
        iou_upInDownOut = iou(np.argmax(logProbs, axis=-1), gtMasks)

        improvedIn = np.logical_and(iou_upIn > iou_orig, unconverged)
        cIn[improvedIn] += step
        indexer *= False
        indexer[improvedIn] = gtMasks[improvedIn] < 0.5
        logProbs[..., 1][indexer] += step

        improvedOut = np.logical_and(
            np.logical_not(improvedIn),
            np.logical_and(iou_downOut > iou_orig, unconverged))
        cOut[improvedOut] += step
        indexer *= False
        indexer[improvedOut] = gtMasks[improvedOut] < 0.5
        logProbs[..., 1][indexer] -= step

        improvedInOut = np.logical_and(
            np.logical_not(improvedIn + np.logical_and(
                iou_downOut > iou_orig, unconverged)),
            unconverged)
        cIn[improvedInOut] += step
        cOut[improvedInOut] += step

        if ind is not None and unconverged[ind]:
            print("Iter %02d: iouIn=%.2f iouOut=%.2f iouInOut=%.2f " % (i,
                    iou_upIn[ind], iou_downOut[ind], iou_upInDownOut[ind],) +
                "iouOrig=%.2f cIn=%.1f cOut=%.1f" % (iou_orig[ind], cIn[ind],
                    cOut[ind]))

    if ind is not None:
        print('iou at convergence=%.2f' % iou_orig[ind])
    newMasks = np.argmax(logProbs, axis=-1).astype(np.int32)
    return newMasks


def iou(mask1, mask2):
    axis = (1, 2) if len(mask1.shape) > 2 else None
    union = np.sum(np.logical_or(mask1, mask2), axis=axis).astype('float')
    if axis is None and union == 0:
        return 0.0
    elif axis is not None:
        union[union == 0] += 1e-12
    intersection = np.sum(
        np.logical_and(mask1, mask2), axis=axis).astype('float')
    return intersection/union


def save_checkpoint(state, is_best, file_path='', filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(file_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(file_path, filename), os.path.join(file_path, 'model_best.pth.tar'))


parser = argparse.ArgumentParser(description='PyTorch DeepMask/SharpMask Training')
parser.add_argument('--rundir', default='./exps/', help='experiments directory')
parser.add_argument('--seed', default=1, type=int, help='manually set RNG seed')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--maxload', default=4000, type=int, metavar='N',
                    help='max number of training batches per epoch')
parser.add_argument('--testmaxload', default=500, type=int, metavar='N',
                    help='max number of testing batches')
parser.add_argument('--maxepoch', default=300, type=int, metavar='N',
                    help='max number of training epochs')
parser.add_argument('--iSz', default=160, type=int, metavar='N',
                    help='input size')
parser.add_argument('--oSz', default=56, type=int, metavar='N',
                    help='output size')
parser.add_argument('--gSz', default=112, type=int, metavar='N',
                    help='ground truth size')
parser.add_argument('--shift', default=16, type=int, metavar='N',
                    help='shift jitter allowed')
parser.add_argument('--scale', default=.25, type=float,
                    help='scale jitter allowed')
parser.add_argument('--hfreq', default=.5, type=float,
                    help='mask/score head sampling frequency')
parser.add_argument('--scratch', action='store_true',
                    help='train DeepMask with randomly initialize weights')
parser.add_argument('--km', default=32, type=int, help='km')
parser.add_argument('--ks', default=32, type=int, help='ks')
parser.add_argument('--freeze_bn', action='store_true',
                    help='freeze running statistics in BatchNorm layers during training (default: False)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-v', '--visualize', action='store_true',
                    help='visualize the result heatmap')




def main():
    global args, device, max_acc, writer

    max_acc = -1
    args = parser.parse_args()

    # Setup experiments results path
    pathsv = 'train'  # additional path folder for saving results
    rundir = os.path.join(args.rundir, pathsv)
    try:
        if not os.path.isdir(rundir):
            os.makedirs(rundir)
    except OSError as err:
        print(err)

    # Get argument defaults (hastag #thisisahack)
    parser.add_argument('--IGNORE', action='store_true')
    defaults = vars(parser.parse_args(['--IGNORE']))

    # Print all arguments, color the non-defaults
    for argument, value in sorted(vars(args).items()):
        reset = colorama.Style.RESET_ALL
        color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
        print('{}{}: {}{}'.format(color, argument, value, reset))

    # Setup seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup Model
    Config = collections.namedtuple('Config', ['iSz', 'oSz', 'gSz'])
    deepmask_config = Config(iSz=args.iSz, oSz=args.oSz, gSz=args.gSz)
    model = DeepMask(deepmask_config).to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # Setup data loader
    train_dataset = get_loader(args.dataset)(args, split='train')
    val_dataset = get_loader(args.dataset)(args, split='val')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch, num_workers=args.workers,
        pin_memory=True, sampler=None)

    # Setup Metrics
    criterion = torch.nn.SoftMarginLoss().to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 120], gamma=0.3)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            max_acc = checkpoint['max_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.maxepoch):
        scheduler.step(epoch=epoch)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % 2 == 1:
            acc = validate(val_loader, model, criterion, epoch)

            is_best = acc > max_acc
            max_acc = max(acc, max_acc)
            # remember best mean loss and save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'max_acc': max_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, rundir)


if __name__ == '__main__':
    main()