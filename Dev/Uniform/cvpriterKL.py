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
import dataloaderiter as dload
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
            netD = models.Discriminator(in_channels=6, n_layers=3, istest=True, ndf=ndf)
    else:   
            netD = models.Discriminator(in_channels=3, n_layers=3, istest=True, ndf=ndf)
    netG = models.CEGeneratorP(in_channels=3, numdowns=depth, istest=True, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #

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
        netD = models.Discriminator(in_channels=6, n_layers =1 , ndf=ndf)##netG = models.CEGenerator(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    else:
    	netD = models.Discriminator(in_channels=3, n_layers =1 , ndf=ndf)
    	#netG = models.UnetGenerator(in_channels=3, num_downs =depth, ngf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    	#netD = models.Discriminator(in_channels=6, n_layers =depth-1, ndf=ngf/4)
    	netEn = models.Encoder(in_channels=3, n_layers =depth, ndf=ngf)
    	netDe = models.Decoder(in_channels=3, n_layers =depth, ndf=ngf)

    # Initialize parameters
    models.network_init(netEn, ctx=ctx)
    models.network_init(netDe, ctx=ctx)
    models.network_init(netD, ctx=ctx)
    if solver=='adam':
            # trainer for the generator and the discriminato
		trainerEn = gluon.Trainer(netEn.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
		trainerDe = gluon.Trainer(netDe.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
		trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    elif solver == 'sgd':
            print('sgd')
            trainerG = gluon.Trainer(netG.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9} )
            trainerD = gluon.Trainer(netD.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9})
    return netEn, netDe, netD, trainerEn, trainerDe, trainerD

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def train(pool_size, epochs, train_data, val_data,  ctx, netEn, netDe,  netD, trainerEn, trainerDe, trainerD, lambda1, batch_size, expname, append=True, useAE = False):
    
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    #netGT, netDT, _, _ = set_test_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf, opt.append)
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    image_pool = imagePool.ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)
    metric2 = mx.metric.MSE()
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    for epoch in range(epochs):

        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        #print('learning rate : '+str(trainerD.learning_rate ))
	for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)
            soft_zero = 1e-10
            fake_latent= netEn(real_in)
	    fake_latent = np.squeeze(fake_latent)
            mu_lv = nd.split(fake_latent, axis=1, num_outputs=2)
	    mu = (mu_lv[0])
            lv = (mu_lv[1])
	    KL = 0.5*nd.nansum(1+lv-mu*mu-nd.exp(lv+soft_zero))
            eps = nd.random_normal(loc=0, scale=1, shape=(batch_size, 2048), ctx=ctx)
            z = mu + nd.exp(0.5*lv)*eps
	    z = nd.expand_dims(nd.expand_dims(z,2),2)
            y = netDe(z)
            fake_out = y
	    
	    logloss = nd.nansum(real_in*nd.log(y+soft_zero)+ (1-real_in)*nd.log(1-y+soft_zero))
            loss = -logloss-KL
            fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history imagesi
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

                fake_latent= np.squeeze(netEn(real_in))
                mu_lv = nd.split(fake_latent, axis=1, num_outputs=2)
                mu = mu_lv[0]
                lv = mu_lv[1]
		KL = 0.5*nd.nansum(1+lv-mu*mu-nd.exp(lv+soft_zero))
                eps = nd.random_normal(loc=0, scale=1, shape=(batch_size, 2048), ctx=ctx)
		#KL = 0.5*nd.nansum(1+lv-mu*mu-nd.exp(lv+soft_zero))
                z = mu + nd.exp(0.5*lv)*eps
		z = nd.expand_dims(nd.expand_dims(z,2),2)
                y = netDe(z)
                fake_out = y
		logloss = nd.nansum((real_in+1)*0.5*nd.log(0.5*(y+1)+soft_zero)+ (1-0.5*(real_in+1))*nd.log(1-0.5*(y+1)+soft_zero))
                loss =-logloss-KL
                fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + loss*lambda1 #L1_loss(real_out, fake_out) * lambda1
                errR = logloss#L1_loss(real_out, fake_out)
                errG.backward()
        trainerDe.step(batch.data[0].shape[0])	   
        trainerEn.step(batch.data[0].shape[0])
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
		logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f reconstruction error= %f at iter %d epoch %d'
                    	% (nd.mean(errD).asscalar(),
                      	nd.mean(errG).asscalar(), acc,nd.mean(errR).asscalar() ,iter, epoch))
        iter = iter + 1
        btic = time.time()

        name, acc = metric.get()
        metric.reset()
        train_data.reset()

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            text_file = open(expname + "_validtest.txt", "a")
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            fake_img1 = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img4 = nd.concat(real_in[3],real_out[3], fake_out[3], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                
            	real_in = vbatch.data[0].as_in_context(ctx)
            	real_out = vbatch.data[1].as_in_context(ctx)

            	fake_latent= netEn(real_in)
            	mu_lv = nd.split(fake_latent, axis=1, num_outputs=2)
            	mu = mu_lv[0]
            	lv = mu_lv[1]
            	eps = nd.random_normal(loc=0, scale=1, shape=(batch_size/5, 2048,1,1), ctx=ctx)
            	z = mu + nd.exp(0.5*lv)*eps
            	y = netDe(z)
            	fake_out = y
            	KL = 0.5*nd.sum(1+lv-mu*mu-nd.exp(lv),axis=1)
            	logloss = nd.sum(real_in*nd.log(y+soft_zero)+ (1-real_in)*nd.log(1-y+soft_zero), axis=1)
            	loss = logloss+KL
            	metric2.update([fake_out, ], [real_out, ])
            	_, acc2 = metric2.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metric2.reset()

            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            #fake_img4T = nd.concat(real_in[3],real_out[3], fake_out[3], dim=1)
            fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img1T,fake_img2T, fake_img3T,dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
            text_file.close()
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
    if opt.useAE == 1:
        useAE = True
    else:
        useAE = False
    if opt.seed != -1:
            random.seed(opt.seed)
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    inclasspaths , inclasses = dload.loadPaths(opt.dataset, opt.datapath, opt.expname, opt.batch_size+1, opt.classes)
    train_data, val_data = load_image.load_image(inclasspaths, opt.batch_size, opt.img_wd, opt.img_ht, opt.noisevar)
    print('Data loading done.')

    if opt.istest:
        testclasspaths = []
        testclasslabels = []
        if opt.istest:
            filename = '_testlist.txt'
        elif opt.isvalidation:
            filename = '_trainlist.txt'
        else:
            filename = '_validationlist.txt'
            filename = '_trainlist.txt'
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


        return ([0,0,0,0])


    else:
        netEn, netDe,  netD, trainerEn, trainerDe, trainerD = set_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf, opt.append)
        if opt.graphvis:
            print(netG)
        print('training')
        print(opt.epochs)
        loss_vec = train(opt.pool_size, opt.epochs, train_data,val_data, ctx, netEn, netDe,  netD, trainerEn, trainerDe, trainerD, opt.lambda1, opt.batch_size, opt.expname,  opt.append, useAE = useAE)
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
