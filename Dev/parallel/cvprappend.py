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


def set_test_network(depth, ctx, lr, beta1, ndf,ngf, append=True):
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









def set_network(depth, ctx, lr, beta1, ndf, ngf, append=True, solver='adam'):
    # Pixel2pixel networks
    if append:
        netD = models.Discriminator(in_channels=6, n_layers =depth-1, ndf=ndf)##netG = models.CEGenerator(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    else:
	netD = models.Discriminator(in_channels=3, n_layers =depth-1, ndf=ndf)
    netG = models.CEGenerator(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    #netD = models.Discriminator(in_channels=6, n_layers =depth-1, ndf=ngf/4)

    # Initialize parameters
    models.network_init(netG, ctx=ctx)
    models.network_init(netD, ctx=ctx)
    if solver=='adam':
	    # trainer for the generator and the discriminator
	    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
	    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    elif solver == 'sgd':
	    print('sgd')
 	    trainerG = gluon.Trainer(netG.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9} )
	    trainerD = gluon.Trainer(netD.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9})
    return netG, netD, trainerG, trainerD

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def train(pool_size, epochs, train_data, ctx, netG, netD, trainerG, trainerD, lambda1, batch_size, expname, append=True):

    netGT, netDT, _, _ = set_test_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf, opt.append)
    dlr = trainerD.learning_rate 
    glr = trainerG.learning_rate     
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    image_pool = imagePool.ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    print(epochs)
    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
	if epoch>250:
 		trainerD.set_learning_rate(dlr * (1-int(epoch-250)/1000))
        	trainerG.set_learning_rate(glr * (1-int(epoch-250)/1000))
        #print('learning rate : '+str(trainerD.learning_rate ))
	for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)

            fake_out = netG(real_in)
            fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history images
                output = netD(fake_concat)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label, ], [output, ])
                real_concat =  nd.concat(real_in, real_out, dim=1) if append else  real_out
                output = netD(real_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                metric.update([real_label, ], [output, ])

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out = netG(real_in)
                fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                errR = L1_loss(real_out, fake_out)
                errG.backward()
	   
            trainerG.step(batch.data[0].shape[0])
            loss_rec_G.append(nd.mean(errG).asscalar()-nd.mean(errR).asscalar()*lambda1)
            loss_rec_D.append(nd.mean(errD).asscalar())
            loss_rec_R.append(nd.mean(errR).asscalar())
            name, acc = metric.get()
	    acc_rec.append(acc)
            # Print log infomation every ten batches
            if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                #print(errD)
		logging.info(
                    'discriminator loss = %f, generator loss = %f, binary training acc = %f reconstruction error= %f at iter %d epoch %d'
                    % (nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), acc,nd.mean(errR).asscalar() ,iter, epoch))
            iter = iter + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_G.params"
            netG.save_params(filename)
   	    netGT.load_params('checkpoints/'+opt.expname+'_'+str(epoch)+'_G.params', ctx=ctx)
    	    netDT.load_params('checkpoints/'+opt.expname+'_'+str(epoch)+'_D.params', ctx=ctx)
            # Visualize one generated image for each epoch
            fake_img1 = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img4 = nd.concat(real_in[3],real_out[3], fake_out[3], dim=1)
	    fake_out = netGT(real_in)
            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img4T = nd.concat(real_in[3],real_out[3], fake_out[3], dim=1)

            fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4,fake_img1T,fake_img2T, fake_img3T,fake_img4T ,dim=2)
            #print(np.shape(fake_img))
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
    return([loss_rec_D,loss_rec_G, loss_rec_R, acc_rec])


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

def main(opt):   
    if opt.seed != -1:
            random.seed(opt.seed)
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    inclasspaths , inclasses = dload.loadPaths(opt.dataset, opt.datapath, opt.expname, opt.batch_size+1, opt.classes)
    train_data, val_data = load_image.load_image(inclasspaths, opt.batch_size, opt.img_wd, opt.img_ht, opt.noisevar)
    print('Data loading done.')
    netG, netD, trainerG, trainerD = set_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf, opt.append)
    if opt.graphvis:
        print(netG)
    print('training')
    print(opt.epochs)
    loss_vec = train(opt.pool_size, opt.epochs, train_data, ctx, netG, netD, trainerG, trainerD, opt.lambda1, opt.batch_size, opt.expname,  opt.append)
    plt.gcf().clear()
    plt.plot(loss_vec[0], label="D", alpha = 0.7)
    plt.plot(loss_vec[1], label="G", alpha=0.7)
    plt.plot(loss_vec[2], label="R", alpha= 0.7)
    plt.plot(loss_vec[3], label="Acc", alpha = 0.7)
    plt.legend()
    plt.savefig('outputs/'+opt.expname+'_loss.png')
    return inclasses

if __name__ == "__main__":
    opt = options.train_options()     
    inclasses = main(opt)
    text_file1 = open("classnames.txt","a")
    text_file1.write("%s \n" % (inclasses[0]))
    text_file1.close()    
#print("Training finished for "+ inclasses)
