#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

if (sys.path[0] + "/netrt:" not in os.environ['LD_LIBRARY_PATH']):
    os.environ['LD_LIBRARY_PATH'] = sys.path[0] + "/netrt:" + os.environ['LD_LIBRARY_PATH']
    os.execv(sys.argv[0], sys.argv)

from netrt import *
import argparse


def gen_net(args):
    net = MixNet(args.prototxt)
    assert net.setMaxBatch(args.max_batch_size)
    assert net.prepareToRun()
    return net


def test_net(net, save_name, batch_size=1, times=1, write=False):
    batch_size = min(net.getMaxBatch(), batch_size)
    net.forward(batch_size)

    total_cost = 0
    for i in range(0, times):
        batch, cost = net.forward(batch_size)
        print("run batch: %d cost: %.2f ms" % (batch, cost / 1000.0))
        total_cost += cost

    print("---------------------------")
    print("avg cost: %.2f ms" % (total_cost / 1000.0 / times))
    if write:
        with open('./log.txt', 'a+') as fw:
            fw.write('{} avg cost: {}ms\n'.format(save_name, (total_cost / 1000.0 / times)))


def main():
    parser = argparse.ArgumentParser()

    # fp32 fp16 int8 caffe
    parser.add_argument("prototxt", help="rel.prototxt | engine.bin", action=None)
    parser.add_argument("model", nargs="?", default=None, help="model.bin (needed when test INT8)")
    parser.add_argument("--write", default=False, action='store_true')
    parser.add_argument("-t", "--type", help="choose test type", action=None, default="FP32", choices=["FP32", "FP16", "MIXFP16", "INT8", "CAFFE", "ENGINE"])
    parser.add_argument("-b", "--max_batch_size", type=int, help="set max batch size", default=1)
    parser.add_argument("-B", "--test_batch_size", type=int, help="set test batch size (default: max_batch_size)", default=-1)
    parser.add_argument("-n", "--num", type=int, help="set test times", default=1)
    parser.add_argument("-save", "--save_name", type=str, help="save name", default=None)
    args = parser.parse_args()

    net = gen_net(args)

    batch_size = args.test_batch_size
    if (args.test_batch_size == -1):
        batch_size = args.max_batch_size

    print("Test forward cost for %s mode:" % args.type)
    test_net(net, args.save_name, batch_size, args.num, args.write)


if __name__ == "__main__":
    main()
