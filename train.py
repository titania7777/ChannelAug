from __future__ import print_function

import argparse
import time
import numpy as np

from models.ResNeXt_DenseNet.models.densenet import densenet
from models.ResNeXt_DenseNet.models.resnext import resnext29
from models.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from ChannelAug import ChannelSplit, ChannelSplit2, ChannelMix
from matplotlib import pyplot as plt
from utils import nentr


# Code From https://github.com/mlaves/bayesian-temperature-scaling
def uceloss(softmaxes, labels, n_bins=15):
    d = softmaxes.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    _, predictions = torch.max(softmaxes, 1)
    errors = predictions.ne(labels)
    uncertainties = nentr(softmaxes, base=softmaxes.size(1))
    errors_in_bin_list = []
    avg_entropy_in_bin_list = []

    uce = torch.zeros(1, device=d)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate |uncert - err| in each bin
        in_bin = uncertainties.gt(bin_lower.item()) * uncertainties.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        if prop_in_bin.item() > 0.0:
            errors_in_bin = errors[in_bin].float().mean()  # err()
            avg_entropy_in_bin = uncertainties[in_bin].mean()  # uncert()
            uce += torch.abs(avg_entropy_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_entropy_in_bin_list.append(avg_entropy_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=d)
    avg_entropy_in_bin = torch.tensor(avg_entropy_in_bin_list, device=d)

    return uce, err_in_bin, avg_entropy_in_bin

# Code From https://github.com/mlaves/bayesian-temperature-scaling
def eceloss(softmaxes, labels, n_bins=15):
    """
    Modified from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """
    d = softmaxes.device
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=d)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    accuracy_in_bin_list = []
    avg_confidence_in_bin_list = []

    ece = torch.zeros(1, device=d)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            accuracy_in_bin_list.append(accuracy_in_bin)
            avg_confidence_in_bin_list.append(avg_confidence_in_bin)

    acc_in_bin = torch.tensor(accuracy_in_bin_list, device=d)
    avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=d)

    return ece, acc_in_bin, avg_conf_in_bin


parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--corruption_path', type=str, default='./data/cifar/', help='Corruption dataset path.')
parser.add_argument('--model', '-m', type=str, default='wrn', choices=['wrn', 'allconv', 'densenet', 'resnext'], help='Choose models.')
parser.add_argument('--epochs', '-e', type=int, default=500, help='Epochs.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-wd', type=float, default=0.0005, help='Weight decay.')
parser.add_argument('--print-freq', type=int, default=50, help='Training loss print frequency.')
parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads.')
args = parser.parse_args()

def train(model, train_loader, optimizer, scheduler):
    model.train()
    loss_ema = 0.
    for i, (images, targets) in enumerate(train_loader):
      optimizer.zero_grad()
      images = images.cuda()
      targets = targets.cuda()
      logits = model(images)
      loss = F.cross_entropy(logits, targets)
      
      loss.backward()
      optimizer.step()
      scheduler.step()
      loss_ema = loss_ema * 0.1 + float(loss) * 0.9
      if i % args.print_freq == 0:
          print('Train Loss {:.3f}'.format(loss_ema))
    return loss_ema

# Code From https://github.com/mlaves/bayesian-temperature-scaling
def plot_conf(acc, conf):
    fig, ax = plt.subplots(1, 1, figsize=(2.5*2, 2.25*2))
    ax.plot([0,1], [0,1], 'k--')
    ax.plot(conf.data.cpu().numpy(), acc.data.cpu().numpy(), marker='.')
    ax.set_xlabel(r'confidence', fontsize=16)
    ax.set_ylabel(r'accuracy', fontsize=16)
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax
# Code From https://github.com/mlaves/bayesian-temperature-scaling
def plot_uncert(err, entr):
    fig, ax = plt.subplots(1, 1, figsize=(2.5*2, 2.25*2))
    ax.plot([0,1], [0,1], 'k--') 
    ax.plot(entr.data.cpu().numpy(), err.data.cpu().numpy(), marker='.')
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_ylabel(r'error', fontsize=16)
    ax.set_xlabel(r'uncertainty', fontsize=16)
    ax.set_xticks((np.arange(0, 1.1, step=0.2)))
    ax.set_yticks((np.arange(0, 1.1, step=0.2)))

    return fig, ax
# Code From https://github.com/mlaves/bayesian-temperature-scaling
def calibration(model, test_loader, save=False, title=''):
    logits = []
    labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            output = model(images)
            logits.append(torch.softmax(output, dim=1).detach())
            labels.append(targets.detach())
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        ece, acc, conf = eceloss(logits, labels)
        uce, err, entr = uceloss(logits, labels)
        print('Test ECE : {:.2f}, UCE : {:.2f}'.format(ece.item()*100, uce.item()*100))
        
    if save:
        fig1, ax1 = plot_conf(acc, conf)
        fig2, ax2 = plot_uncert(err, entr)
        
        textstr1 = r'ECE={:.2f}'.format(ece.item()*100)
        props = dict(boxstyle='round', facecolor='white', alpha=0.75)
        ax1.text(0.075, 0.925, textstr1, transform=ax1.transAxes, fontsize=14,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=props)
        ax1.set_title(title.replace('_', ' '), fontsize=16)
        fig1.tight_layout()
        fig1.savefig('ECE/' + title + '_ECE.png')

        textstr2 = r'UCE={:.2f}'.format(uce.item()*100)
        ax2.text(0.925, 0.075, textstr2, transform=ax2.transAxes, fontsize=14,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=props)
        ax2.set_title(title.replace('_', ' '), fontsize=16)
        fig2.tight_layout()
        fig2.savefig('UCE/' + title + '_UCE.png')

def get_lr(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def test(model, test_loader, calibration=False,):
    model.eval()
    total_loss = 0.
    total_correct = 0
    logits_ = []
    labels_ = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()
            if calibration:
                logits_.append(torch.softmax(logits, dim=1).detach())
                labels_.append(targets.detach())
    if calibration:
        logits_ = torch.cat(logits_, dim=0)
        labels_ = torch.cat(labels_, dim=0)
        ece, acc, conf = eceloss(logits_, labels_)
        uce, err, entr = uceloss(logits_, labels_)
        return ece, uce, total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)
    return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)

def test_c(net, test_data, base_path):
    corruption_accs = []
    ece_c = 0
    uce_c = 0

    CORRUPTIONS = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
    ]

    for corruption in CORRUPTIONS:
        # Reference to original data is mutated
        test_data.data = np.load(base_path + corruption + '.npy')
        test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1000,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True)

        ece, uce, test_loss, test_acc = test(net, test_loader, True)
        corruption_accs.append(test_acc)
        ece_c += ece.item()*100
        uce_c += uce.item()*100
        
        print('{}: Test Loss {:.3f} | Test Error {:.3f} | ECE : {:.2f} | UCE : {:.2f}'.format(corruption, test_loss, 100 - 100. * test_acc, ece.item()*100, uce.item()*100))
    print('[Mean Corruption ECE : {:.2f}, UCE : {:.2f}]'.format(ece_c/15, uce_c/15))
    return np.mean(corruption_accs)


def main():
    torch.manual_seed(7777)
    np.random.seed(7777)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        ChannelMix(skip=False, sum=False, prob=0.7, beta=5, width=3),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=transform_test, download=True)
        base_c_path = args.corruption_path + '/CIFAR-10-C/'
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data/cifar', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data/cifar', train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        test_data = datasets.CIFAR10(
            './data/cifar', train=False, transform=transform_test, download=True)
        base_c_path = args.corruption_path + '/CIFAR-100-C/'
        num_classes = 100

    #models
    if args.model == 'densenet':
        model = densenet(num_classes=num_classes)
    elif args.model == 'wrn':
        model = WideResNet(40, num_classes, 2, 0.0)
    elif args.model == 'resnext':
        model = resnext29(num_classes=num_classes)

    cudnn.benchmark = True
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, args.epochs * len(train_loader), 1, 1e-6 / args.learning_rate))

    for epoch in range(args.epochs):
        begin_time = time.time()
        train_loss_ema = train(model, train_loader, optimizer, scheduler)
        test_loss, test_acc = test(model, test_loader)
        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format((epoch + 1), int(time.time() - begin_time), train_loss_ema, test_loss, 100 - 100. * test_acc))
    calibration(model, test_loader, True, args.dataset)
    test_c_acc = test_c(model, test_data, base_c_path)
    print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))

if __name__ == '__main__':
    main()