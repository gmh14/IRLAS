# -*- coding: utf-8 -*-
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import numpy as np
import argparse

from ImageNetDataset import ImageDataset
import utils


class Env(object):
    def __init__(self, args, env_id, save_dir):
        self.save = save_dir
        self.env = env_id
        self.workers = args.workers
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.dataset_path = args.dataset_path
        self.logger = utils.create_logger('global_logger', self.save + '/log.txt')

    def get_gpumodel(self, model):
        model = model.cuda()
        return model

    def get_loss_func(self):
        return nn.CrossEntropyLoss().cuda()

    def validate(self, val_loader, model, criterion):
        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda()
                input = input.cuda()

                # compute output
                output = model(input)

                # measure accuracy and record loss
                loss = criterion(output, target)
                prec1, prec5 = utils.accuracy(output, target, topk=(1, 5))
                losses.update(loss.clone().item(), input.size(0))
                top1.update(prec1.clone().item(), input.size(0))
                top5.update(prec5.clone().item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    self.logger.info('Test: [{0}/{1}]\t'
                                     'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                     'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                     'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader), batch_time=batch_time, loss=losses,
                                                                                     top1=top1, top5=top5))

        self.logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        return top1.avg


class Env_ImageNet(Env):
    def __init__(self, args, env_id, save_dir):
        super(Env_ImageNet, self).__init__(args, env_id, save_dir)
        self.input_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_test = transforms.Compose([
            transforms.Resize({224: 256, 299: 333, 331: 367}[self.input_size]),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            normalize,
        ])

        val_dataset = ImageDataset(
            '{}/val'.format(self.dataset_path),
            '{}/meta/val.txt'.format(self.dataset_path),
            transform_test)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)


def get_args():
    env_parser = argparse.ArgumentParser()
    env_parser.add_argument('--dataset_path', type=str, default='/mnt/lustre/share/images', help='data root')
    env_parser.add_argument('-j', '--workers', default=4, type=int,
                            help='number of data loading workers (default: 4)')
    env_parser.add_argument('-b', '--batch_size', default=256, type=int,
                            help='mini-batch size (default: 256)')
    env_parser.add_argument('--print_freq', default=100, type=int,
                            help='print frequency (default: 100)')
    env_parser.add_argument('--model', default='IRLAS', type=str,
                            help='model to validate: IRLAS')
    env_args = env_parser.parse_args()
    return env_args


if __name__ == '__main__':
    env_args = get_args()
    env = Env_ImageNet(env_args, 0, '.')
    model = utils.get_model(env_args.model, True)
    env.logger.info('  Total params: %.2fM' % (model.netpara))
    model = env.get_gpumodel(model)
    prec = env.validate(env.val_loader, model, env.get_loss_func())
    print(prec)
