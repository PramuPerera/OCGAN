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
import options
import visual

import argparse
#logging.basicConfig()



def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()



def set_network(depth, ctx, lr, beta1, ndf,ngf, append=True):
    if append:
            netD = models.Discriminator(in_channels=6, n_layers=depth-1, istest=True, ndf=ndf)
    else:        
            netD = models.Discriminator(in_channels=3, n_layers=depth-1, istest=True, ndf=ndf)
    netG = models.CEGenerator(in_channels=3, n_layers=depth, istest=True, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #

    # Initialize parameters
    models.network_init(netG, ctx=ctx)
    models.network_init(netD, ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    return netG, netD, trainerG, trainerD

def main(opt):
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    testclasspaths = []
    testclasslabels = []
    if opt.istest:
        filename = '_testlist.txt'
    else:
        filename = '_validationlist.txt'        
    with open(opt.dataset+"_"+opt.expname+filename , 'r') as f:
        for line in f:
            testclasspaths.append(line.split(' ')[0])
            if int(line.split(' ')[1])==-1:
                testclasslabels.append(0)
            else:
                testclasslabels.append(1)

    test_data = load_image.load_test_images(testclasspaths,testclasslabels,opt.batch_size, opt.img_wd, opt.img_ht, ctx, opt.noisevar)
    netG, netD, trainerG, trainerD = set_network(opt.depth, ctx, 0, 0, opt.ndf, opt.ngf, opt.append)
    netG.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_G.params', ctx=ctx)
    netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
    lbllist = [];
    scorelist1 = [];
    scorelist2 = [];
    scorelist3 = [];
    scorelist4 = [];
    test_data.reset()
    count = 0
    for batch in (test_data):
        count+=1
        real_in = batch.data[0].as_in_context(ctx)
        real_out = batch.data[1].as_in_context(ctx)
        lbls = batch.label[0].as_in_context(ctx)
        outnn = (netG(real_out))
        out_concat =  nd.concat(real_out, outnn, dim=1) if opt.append else  outnn
        output4 = nd.mean((netD(out_concat)), (1, 3, 2)).asnumpy()    
        out = (netG(real_in))
        out_concat =  nd.concat(real_in, out, dim=1) if opt.append else  out
        output = netD(out_concat) #Denoised image
        output3 = nd.mean(out-real_out, (1, 3, 2)).asnumpy() #denoised-real
        output = nd.mean(output, (1, 3, 2)).asnumpy()
        out_concat =  nd.concat(real_out, real_out, dim=1) if opt. append else  real_out
        output2 = netD(out_concat) #Image with no noise
        output2 = nd.mean(output2, (1, 3, 2)).asnumpy()
        lbllist = lbllist+list(lbls.asnumpy())
        scorelist1 = scorelist1+list(output)
        scorelist2 = scorelist2+list(output2)
        scorelist3 = scorelist3+list(output3)
        scorelist4 = scorelist4+list(output4)
	
	fake_img1 = nd.concat(real_in[0],real_out[0], out[0], outnn[0],dim=1)
        fake_img2 = nd.concat(real_in[1],real_out[1], out[1],outnn[1], dim=1)
        fake_img3 = nd.concat(real_in[2],real_out[2], out[2], outnn[2], dim=1)
        fake_img4 = nd.concat(real_in[3],real_out[3],out[3],outnn[3], dim=1)
        fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
        #print(np.shape(fake_img))
        visual.visualize(fake_img)
        plt.savefig('outputs/T_'+opt.expname+'_'+str(count)+'.png')




    fpr, tpr, _ = roc_curve(lbllist, scorelist1, 1)
    roc_auc1 = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(lbllist, scorelist2, 1)
    roc_auc2 = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(lbllist, scorelist3, 1)
    roc_auc3 = auc(fpr, tpr)
    fpr, tpr, _ = roc_curve(lbllist, scorelist4, 1)
    roc_auc4 = auc(fpr, tpr)
    return([roc_auc1, roc_auc2, roc_auc3, roc_auc4])

if __name__ == "__main__":
    opt = options.test_options()
    roc_auc = main(opt)
    print(roc_auc)
