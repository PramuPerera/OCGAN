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







def set_network(depth, ctx, lr, beta1, ndf, ngf,latent, append=True, solver='adam'):
    # Pixel2pixel networks
    if append:
        netD = models.Discriminator(in_channels=6, n_layers =2 , ndf=ndf)##netG = models.CEGenerator(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
        netD2 = models.LatentDiscriminator(in_channels=6, n_layers =2 , ndf=ndf)##netG = models.CEGenerator(in_channels=3, n_layers=depth, ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #

    else:
        netD = models.Discriminator(in_channels=3, n_layers =2 , ndf=ndf)
        netD2 = models.LatentDiscriminator(in_channels=3, n_layers =2 , ndf=ndf)
        #netG = models.UnetGenerator(in_channels=3, num_downs =depth, ngf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
        #netD = models.Discriminator(in_channels=6, n_layers =depth-1, ndf=ngf/4)
        netEn = models.Encoder(in_channels=3, n_layers =depth,latent=latent, ndf=ngf)
        netDe = models.Decoder(in_channels=3, n_layers =depth, latent=latent, ndf=ngf)

    # Initialize parameters
    models.network_init(netEn, ctx=ctx)
    models.network_init(netDe, ctx=ctx)
    models.network_init(netD, ctx=ctx)
    models.network_init(netD2, ctx=ctx)
    
    trainerEn = gluon.Trainer(netEn.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerDe = gluon.Trainer(netDe.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD2 = gluon.Trainer(netD2.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    return netEn, netDe, netD, netD2, trainerEn, trainerDe, trainerD, trainerD2

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def train(pool_size, epochs, train_data, val_data,  ctx, netEn, netDe,  netD, netD2, trainerEn, trainerDe, trainerD, trainerD2, lambda1, batch_size, expname,  append=True, useAE = False):
    tp_file = open(expname + "_trainloss.txt", "w")  
    tp_file.close()  
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    #netGT, netDT, _, _ = set_test_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf, opt.append)
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    image_pool = imagePool.ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)
    metric2 = mx.metric.CustomMetric(facc)
    metricMSE = mx.metric.MSE()
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    acc2_rec = []
    loss_rec_D2 = []
    loss_rec_G2 = []
    lr = 0.002
    #mu = nd.random_normal(loc=0, scale=1, shape=(batch_size/2,64,1,1), ctx=ctx) 
    mu = nd.random.uniform(low= -1, high=1, shape=(batch_size/2,64,1,1),ctx=ctx)
    #mu =  nd.zeros((batch_size/2,64,1,1),ctx=ctx)
    sigma = nd.ones((64,1,1),ctx=ctx)
    mu.attach_grad()
    sigma.attach_grad()    
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
            fake_latent= netEn(real_in)
            #real_latent = nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx)
            real_latent = nd.multiply(nd.power(sigma,2),nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx))
	    #nd.random.uniform( low=-1, high=1, shape=fake_latent.shape, ctx=ctx)
	    fake_out = netDe(fake_latent)
            fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history imagesi
                output = netD(fake_concat)
                output2 = netD2(fake_latent)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                fake_latent_label = nd.zeros(output2.shape, ctx=ctx)
		noiseshape = (fake_latent.shape[0]/2,fake_latent.shape[1],fake_latent.shape[2],fake_latent.shape[3])
                eps2 = nd.multiply(nd.power(sigma,2),nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx))
		#eps2 = nd.random_normal(loc=0, scale=sigma.asscalar(), shape=fake_latent.shape, ctx=ctx) #
		#eps = nd.random.uniform( low=-1, high=1, shape=noiseshape, ctx=ctx)
		rec_output = netD(netDe(eps2))
                errD_fake = GAN_loss(rec_output, fake_label)
                errD_fake2 = GAN_loss(output, fake_label)
                errD2_fake = GAN_loss(output2, fake_latent_label)
                metric.update([fake_label, ], [output, ])
                metric2.update([fake_latent_label, ], [output2, ])
                real_concat =  nd.concat(real_in, real_out, dim=1) if append else  real_out
                output = netD(real_concat)
                output2 = netD2(real_latent)
                real_label = nd.ones(output.shape, ctx=ctx)
                real_latent_label =  nd.ones(output2.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD2_real =  GAN_loss(output2, real_latent_label)
                #errD = (errD_real + 0.5*(errD_fake+errD_fake2)) * 0.5
                errD = (errD_real + errD_fake) * 0.5
                errD2 = (errD2_real + errD2_fake) * 0.5
		totalerrD = errD+errD2
                totalerrD.backward()
                #errD2.backward()
                metric.update([real_label, ], [output, ])
            	metric2.update([real_latent_label, ], [output2, ])
            trainerD.step(batch.data[0].shape[0])
            trainerD2.step(batch.data[0].shape[0])
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
		sh = fake_latent.shape
		eps2 = nd.multiply(nd.power(sigma,2),nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx))
                #eps2 = nd.random_normal(loc=0, scale=sigma.asscalar(), shape=fake_latent.shape, ctx=ctx) #
		#eps = nd.random.uniform( low=-1, high=1, shape=noiseshape, ctx=ctx)
		rec_output = netD(netDe(eps2))
                fake_latent= (netEn(real_in))
                output2 = netD2(fake_latent)
                fake_out = netDe(fake_latent)
                fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                real_latent_label = nd.ones(output2.shape, ctx=ctx)
                errG2 = GAN_loss(rec_output, real_label)
                errR = L1_loss(real_out, fake_out) * lambda1
		errG = 10.0*GAN_loss(output2, real_latent_label)+errG2+errR+nd.mean(nd.power(sigma,2))
		errG.backward()
	    if epoch>50:
	    	sigma -= lr / sigma.shape[0] * sigma.grad
	    	print(sigma)
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
            loss_rec_G2.append(nd.mean(errG2).asscalar())
            loss_rec_G.append(nd.mean(nd.mean(errG)).asscalar()-nd.mean(errG2).asscalar()-nd.mean(errR).asscalar())
            loss_rec_D.append(nd.mean(errD).asscalar())
            loss_rec_R.append(nd.mean(errR).asscalar())
            loss_rec_D2.append(nd.mean(errD2).asscalar())
            _, acc2 = metric2.get()
            name, acc = metric.get()
            acc_rec.append(acc)
            acc2_rec.append(acc2)

            # Print log infomation every ten batches
            if iter % 10 == 0:
                _, acc2 = metric2.get()
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                #print(errD)
		logging.info('discriminator loss = %f, D2 loss = %f, generator loss = %f, G2 loss = %f,  binary training acc = %f , D2 acc = %f, reconstruction error= %f  at iter %d epoch %d'
                    	% (nd.mean(errD).asscalar(),nd.mean(errD2).asscalar(),
                      	nd.mean(errG-errG2-errR).asscalar(),nd.mean(errG2).asscalar(), acc,acc2,nd.mean(errR).asscalar() ,iter, epoch))
                iter = iter + 1
        btic = time.time()
        name, acc = metric.get()
        _, acc2 = metric2.get()
        tp_file = open(expname + "_trainloss.txt", "a")
        tp_file.write(str(nd.mean(errG2).asscalar()) + " " + str(
            nd.mean(nd.mean(errG)).asscalar() - nd.mean(errG2).asscalar() - nd.mean(errR).asscalar()) + " " + str(
            nd.mean(errD).asscalar()) + " " + str(nd.mean(errD2).asscalar()) + " " + str(nd.mean(errR).asscalar()) +" "+str(acc) + " " + str(acc2)+"\n")
        tp_file.close()
        metric.reset()
        metric2.reset()
        train_data.reset()

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:# and epoch>0:
            text_file = open(expname + "_validtest.txt", "a")
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D2.params"
            netD2.save_params(filename)
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
                y = netDe(fake_latent)
                fake_out = y
                metricMSE.update([fake_out, ], [real_out, ])
            _, acc2 = metricMSE.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metricMSE.reset()
	    images = netDe(eps2)
            fake_img1T = nd.concat(images[0],images[1], images[2], dim=1)
	    fake_img2T = nd.concat(images[3],images[4], images[5], dim=1)
            fake_img3T = nd.concat(images[6],images[7], images[8], dim=1)
            fake_img = nd.concat(fake_img1T,fake_img2T, fake_img3T,dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_fakes_'+str(epoch)+'.png')
            text_file.close()

	    # Do 10 iterations of sampler update
	    fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            #fake_img4T = nd.concat(real_in[3],real_out[3], fake_out[3], dim=1)
            fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img1T,fake_img2T, fake_img3T,dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
	    '''if epoch > 100:
	      for ep2 in range(10):
	    	with autograd.record():
                	#eps = nd.random_normal(loc=0, scale=1, shape=noiseshape, ctx=ctx) #
			eps = nd.random.uniform( low=-1, high=1, shape=noiseshape, ctx=ctx)
			eps2 = nd.random_normal(loc=0, scale=0.02, shape=noiseshape, ctx=ctx)
                	eps2 = nd.tanh(eps2*sigma+mu)
                	eps2 = nd.concat(eps,eps2,dim=0)
			rec_output = netD(netDe(eps2))
			fake_label = nd.zeros(rec_output.shape, ctx=ctx)
                	errGS = GAN_loss(rec_output, fake_label)
    			errGS.backward()
		mu -= lr / mu.shape[0] * mu.grad
		sigma -= lr / sigma.shape[0] * sigma.grad
	    	print('mu ' + str(mu[0,0,0,0].asnumpy())+ '  sigma '+ str(sigma[0,0,0,0].asnumpy()))
	    '''
	    images = netDe(eps2)
            fake_img1T = nd.concat(images[0],images[1], images[2], dim=1)
            fake_img2T = nd.concat(images[3],images[4], images[5], dim=1)
            fake_img3T = nd.concat(images[6],images[7], images[8], dim=1)
            fake_img = nd.concat(fake_img1T,fake_img2T, fake_img3T,dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_fakespost_'+str(epoch)+'.png')
    return([loss_rec_D,loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2, loss_rec_G2, acc2_rec])


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
        print('testing not implemented')
    else:
	print("Latent variable dim : " + str(opt.latent))
        netEn, netDe,  netD, netD2, trainerEn, trainerDe, trainerD, trainerD2 = set_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf,opt.latent, opt.append)
        if opt.graphvis:
            print(netG)
        print('training')
        print(opt.epochs)
        loss_vec = train(opt.pool_size, opt.epochs, train_data,val_data, ctx, netEn, netDe,  netD, netD2, trainerEn, trainerDe, trainerD, trainerD2, opt.lambda1, opt.batch_size, opt.expname,  opt.append, useAE = useAE)
        plt.gcf().clear()
	fig, ax1 = plt.subplots()
	ax1.plot(loss_vec[2], label="R", alpha= 0.7)
	ax2 = ax1.twinx() 
        ax2.plot(loss_vec[0], label="Dr", alpha = 0.7)
        ax2.plot(loss_vec[4], label="Dl", alpha = 0.7)
        ax2.plot(loss_vec[1], label="Gr", alpha=0.7)
        ax2.plot(loss_vec[5], label="Gl", alpha=0.7)
        ax2.plot(loss_vec[3], label="Accr", alpha = 0.7)
	ax2.plot(loss_vec[6], label="Accl", alpha = 0.7)
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
