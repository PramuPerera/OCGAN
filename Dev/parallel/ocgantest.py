#https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter14_generative-adversarial-networks/pixel2pixel.ipynb
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
from random import shuffle
from sklearn.metrics import roc_curve, auc
import load_image
import models
from datetime import datetime
import time
import logging



epochs = 200
batch_size = 10
use_gpu = True
ctx = mx.gpu() if use_gpu else mx.cpu()
lr = 0.0002
beta1 = 0.5
lambda1 = 100
pool_size = 50
datapath = '../'
dataset = 'Caltech256'
expname = 'expce'
img_wd = 256
img_ht = 256
testclasspaths = []
testclasslabels = []


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()



def set_network():
    # Pixel2pixel networks
    netG = models.CEGenerator(in_channels=3, istest=True)  # UnetGenerator(in_channels=3, num_downs=8) #
    netD = models.Discriminator(in_channels=6, istest=True)

    # Initialize parameters
    models.network_init(netG, ctx=ctx)
    models.network_init(netD, ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    return netG, netD, trainerG, trainerD



with open(dataset+"_"+expname+"_testlist.txt" , 'r') as f:
    for line in f:
        testclasspaths.append(line.split(' ')[0])
        if int(line.split(' ')[1])==-1:
            testclasslabels.append(0)
        else:
            testclasslabels.append(1)
print(np.shape(testclasslabels))
print(batch_size)
print(np.shape(testclasspaths))
print('Loading data')
test_data = load_image.load_test_images(testclasspaths,testclasslabels,batch_size, img_wd, img_ht, ctx=ctx)
print('Loading Done')

# Loss
GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
L1_loss = gluon.loss.L1Loss()
netG, netD, trainerG, trainerD = set_network()
netG.load_params( "checkpoints/"+expname+"_"+str(epoch)+"_G.params", ctx=ctx)
netD.load_params("checkpoints/"+expname+"_"+str(epoch)+"_D.params", ctx=ctx)
print('Loading model done')

lbllist = [];
scorelist = [];
test_data.reset()
count = 0
print('testing')
for batch in (test_data):
    print(count)
    count+=1
    real_in = batch.data[0].as_in_context(ctx)
    real_out = batch.data[1].as_in_context(ctx)
    lbls = batch.label[0].as_in_context(ctx)
    out = (netG(real_in))
    real_concat = nd.concat(real_in, real_in, dim=1)
    #real_concat = nd.concat(out, out, dim=1)
    output = netD(real_concat)
    output = nd.mean(output, (1, 3, 2)).asnumpy()
    lbllist = lbllist+list(lbls.asnumpy())
    scorelist = scorelist+list(output)
    #visualize(out[0,:,:,:])
    #plt.savefig('outputs/testnet_T' + str(count) + '.png')

print((lbllist))
print((scorelist))
fpr, tpr, _ = roc_curve(lbllist, scorelist, 0)
roc_auc = auc(fpr, tpr)
print(roc_auc)
