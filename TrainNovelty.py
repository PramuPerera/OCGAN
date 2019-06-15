from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
import random
from random import shuffle
import dataloaderiter as dload
import load_image
import visual
import models
from datetime import datetime
import time
import logging
import argparse
import models
import options
import andgan


def plotloss(loss_vec, fname):
    plt.gcf().clear()
    plt.plot(loss_vec[0], label="Dr", alpha = 0.7)
    plt.plot(loss_vec[4], label="Dl", alpha = 0.7)
    plt.plot(loss_vec[1], label="G", alpha=0.7)
    plt.plot(loss_vec[2], label="R", alpha= 0.7)
    plt.plot(loss_vec[3], label="Acc", alpha = 0.7)
    plt.legend()
    plt.savefig(fname)


def main(opt):
    if opt.seed != -1:
        random.seed(opt.seed)
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    inclasspaths , inclasses = dload.loadPaths(opt)
    train_data, val_data = load_image.load_image(inclasspaths, opt)
    print('Data loading done.')
    networks = models.set_network(opt, ctx, False)
    print('training')
    # train networks based on opt.ntype(1 - AE 2 - ALOCC 3 - latentD  4 - adnov)
    if opt.ntype == 4:
        loss_vec = andgan.trainadnov(opt, train_data, val_data, ctx, networks)
    elif opt.ntype == 2:
        loss_vec = andgan.traincvpr18(opt, train_data, val_data, ctx, networks)
    elif opt.ntype == 1:
        loss_vec = andgan.trainAE(opt, train_data, val_data, ctx, networks)
    plotloss(loss_vec, 'outputs/'+opt.expname+'_loss.png')
    return inclasses


if __name__ == "__main__":
    opt = options.train_options()
    inclasses = main(opt)

