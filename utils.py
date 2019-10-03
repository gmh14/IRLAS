from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import re
import hashlib
import logging

from models.IRLAS_mobile import IRLAS

model_path = {
    IRLAS: "pretrained_models/IRLAS-ImageNet-mobile-9.96M-75.15-8804bd1962.pth",
}


def load_model(model, path, check=True):
    if check:
        HASH_REGEX = re.compile(r'-([a-f0-9]*)\.pth')
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            hasher.update(f.read())
        digest = hasher.hexdigest()
        hash_prefix = HASH_REGEX.search(path).group(1)
        if digest[:len(hash_prefix)] != hash_prefix:
            raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                               .format(hash_prefix, digest))
    state_dict = torch.load(path, map_location=None)
    model.load_state_dict(state_dict)
    return model


def get_model(model_type, pretrained=False):
    if model_type == 'IRLAS_mobile':
        model = IRLAS()
    # elif model_type == 'IRLNet_large':
    #     model = IRLNet_large()
    if pretrained:
        model = load_model(model, model_path[model.__class__], check=True)
    return model


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count