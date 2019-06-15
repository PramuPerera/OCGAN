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
import random
from random import shuffle
import dataloader as dload
import load_image
import visual
import models
import imagePool
from datetime import datetime
import time
import logging
import argparse
import options
#logging.basicConfig()

def set_network(depth, ctx, lr, beta1, ngf):
    # Pixel2pixel networks
    #netG = models.CEGenerator(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netEn = models.Encoder(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netDe = models.Decoder(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netD = models.Discriminator(in_channels=3, n_layers =depth, ndf=ngf, isthreeway=True)

    # Initialize parameters
    models.network_init(netDe, ctx=ctx)    
    models.network_init(netEn, ctx=ctx)
    models.network_init(netD, ctx=ctx)

    # trainer for the generator and the discriminator
    trainerEn = gluon.Trainer(netEn.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerDe = gluon.Trainer(netDe.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    return netEn, netDe, netD, trainerEn, trainerDe, trainerD

def facc(label, pred):
    return np.mean((np.argmax(pred,1)== label))

def train(pool_size, epochs, train_data, ctx, netEn, netDe, netD, trainerEn, trainerDe, trainerD, lambda1, batch_size, expname):

    threewayloss =gluon.loss.SoftmaxCrossEntropyLoss()
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L1Loss()
    image_pool = imagePool.ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)

    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)
            tempout = netEn(real_in)
            fake_out = netDe(tempout)
            fake_concat = fake_out
            #fake_concat = image_pool.query(fake_out)
            #fake_concat = image_pool.query(nd.concat(real_in, fake_out, dim=1))
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history images
                output = netD(fake_concat)
                fake_label = nd.zeros(output.shape[0], ctx=ctx)
                errD_fake = threewayloss(output, fake_label)
                metric.update([fake_label, ], [output, ])

                

                # Train with real image
                real_concat = real_out
                output = netD(real_concat)
                real_label = nd.ones(output.shape[0], ctx=ctx)
                errD_real = threewayloss(output, real_label)
                metric.update([real_label, ], [output, ])


                #train with abnormal image
                abinput = nd.random.uniform(-1,1,tempout.shape,ctx=ctx)
                aboutput =netD( netDe(abinput))
		#print(aboutput.shape)
		#print(output.shape)
                ab_label = 2*nd.ones(aboutput.shape[0], ctx=ctx)
                errD_ab = threewayloss(aboutput, ab_label)
                errD = (errD_real + errD_fake + errD_ab) * 0.33
                errD.backward()
                

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out = netDe(netEn(real_in))
                fake_concat = fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape[0], ctx=ctx)
                errG = threewayloss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                errR = L1_loss(real_out, fake_out)
                errG.backward()

            trainerEn.step(batch.data[0].shape[0])
            trainerDe.step(batch.data[0].shape[0])

            # Print log infomation every ten batches
            if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info(
                    'discriminator loss = %f, generator loss = %f, latent error = %f,  binary training acc = %f, reconstruction error= %f at iter %d epoch %d'
                    % (nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), nd.mean(errD_ab).asscalar()   , acc,nd.mean(errR).asscalar() ,iter, epoch))
            iter = iter + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            # Visualize one generated image for each epoch
            fake_img = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
        #


def print_result():
    num_image = 4
    img_in_list, img_out_list = val_data.next().data
    for i in range(num_image):
        img_in = nd.expand_dims(img_in_list[i], axis=0)
        plt.subplot(2, 4, i + 1)
        visual.visualize(img_in[0])
        img_out = netG(img_in.as_in_context(ctx))
        plt.subplot(2, 4, i + 5)
        visual.visualize(img_out[0])
    plt.show()


opt = options.train_options()        
if opt.seed != -1:
	random.seed(opt.seed)
ctx = mx.gpu() if opt.use_gpu else mx.cpu()
inclasspaths, _ = dload.loadPaths(opt.dataset, opt.datapath, opt.expname, opt.batch_size)
train_data, val_data = load_image.load_image(inclasspaths, opt.batch_size, opt.img_wd, opt.img_ht, opt.noisevar)
print('Data loading done.')
netEn, netDe, netD, trainerEn, trainerDe, trainerD = set_network(opt.depth, ctx, opt.lr, opt.beta1, opt.ngf)
if opt.graphvis:
    print(netEn)
print('training')
train(opt.pool_size, opt.epochs, train_data, ctx, netEn, netDe, netD, trainerEn, trainerDe, trainerD, opt.lambda1, opt.batch_size, opt.expname)

