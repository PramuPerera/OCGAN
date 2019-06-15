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
	netDS = models.Discriminator(in_channels=3, n_layers =2 , ndf=64)
        #netG = models.UnetGenerator(in_channels=3, num_downs =depth, ngf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
        #netD = models.Discriminator(in_channels=6, n_layers =depth-1, ndf=ngf/4)
        netEn = models.Encoder(in_channels=3, n_layers =depth,latent=latent, ndf=ngf)
        netDe = models.Decoder(in_channels=3, n_layers =depth, latent=latent, ndf=ngf)

    # Initialize parameters
    models.network_init(netEn, ctx=ctx)
    models.network_init(netDe, ctx=ctx)
    models.network_init(netD, ctx=ctx)
    models.network_init(netD2, ctx=ctx)
    models.network_init(netDS, ctx=ctx)
    trainerEn = gluon.Trainer(netEn.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerDe = gluon.Trainer(netDe.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD2 = gluon.Trainer(netD2.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerDS = gluon.Trainer(netDS.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    return netEn, netDe, netD, netD2,netDS, trainerEn, trainerDe, trainerD, trainerD2, trainerDS

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def train(cep , pool_size, epochs, train_data, val_data,  ctx, netEn, netDe,  netD, netD2, netDS, trainerEn, trainerDe, trainerD, trainerD2, trainerSD, lambda1, batch_size, expname,  append=True, useAE = False):
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
    metricStrong = mx.metric.CustomMetric(facc)
    metricMSE = mx.metric.MSE()
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    acc2_rec = []
    loss_rec_D2 = []
    loss_rec_G2 = []
    lr = 2.0*512
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    if cep == -1:
	cep=0
    else:
	netEn.load_params('checkpoints/'+opt.expname+'_'+str(cep)+'_En.params', ctx=ctx)
    	netDe.load_params('checkpoints/'+opt.expname+'_'+str(cep)+'_De.params', ctx=ctx)
    	netD.load_params('checkpoints/'+opt.expname+'_'+str(cep)+'_D.params', ctx=ctx)
    	netD2.load_params('checkpoints/'+opt.expname+'_'+str(cep)+'_D2.params', ctx=ctx)
        netDS.load_params('checkpoints/'+opt.expname+'_'+str(cep)+'_SD.params', ctx=ctx)
    iter = 0
    for epoch in range(cep+1, epochs):

        tic = time.time()
        btic = time.time()
        train_data.reset()
        #print('learning rate : '+str(trainerD.learning_rate ))
        for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
	    if ctx==mx.cpu():
		ct=mx.cpu()
	    else:
		ct=mx.gpu()
            real_in = batch.data[0]#.as_in_context(ctx)
            real_out = batch.data[1]#.as_in_context(ctx)
	    if iter ==0:
	    	latent_shape = (batch_size,512,1,1) #code.shape
	    	out_l_shape = (batch_size,1,1,1)#netD2((code)).shape
	    	out_i_shape = (batch_size,1,1,1)#netD(netDe(code)).shape
	    	out_s_shape = (batch_size,1,1,1)#netSD(netDe(code)).shape
            real_in  = gluon.utils.split_and_load(real_in, ctx)
	    real_out  = gluon.utils.split_and_load(real_out, ctx)
	    fake_latent= [netEn(r) for r  in real_in]
            real_latent = nd.random.uniform( low=-1, high=1, shape=latent_shape)
	    real_latent = gluon.utils.split_and_load(real_latent, ctx)
	    fake_out = [netDe(f) for f in fake_latent] 
            fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
	    eps2 = nd.random.uniform( low=-1, high=1, shape=latent_shape, ctx=ct)
	    eps2 = gluon.utils.split_and_load(eps2,ctx)
            if epoch > 150: # (1/float(batch_size))*512*150:# and epoch%10==0:
	      print('Mining..')
	      mu = nd.random.uniform(low= -1, high=1, shape=latent_shape,ctx=ct)
    	      #isigma = nd.ones((batch_size,64,1,1),ctx=ctx)*0.000001
	      mu.attach_grad()
	      #sigma.attach_grad()
	      images = netDe(mu)
              fake_img1T = nd.concat(images[0],images[1], images[2], dim=1)
              fake_img2T = nd.concat(images[3],images[4], images[5], dim=1)
              fake_img3T = nd.concat(images[6],images[7], images[8], dim=1)
              fake_img = nd.concat(fake_img1T,fake_img2T, fake_img3T,dim=2)
              visual.visualize(fake_img)
              plt.savefig('outputs/'+expname+'_fakespre_'+str(epoch)+'.png')
	      eps2 = gluon.utils.split_and_load(mu,ctx)
	      for e in eps2:
                         e.attach_grad()
              for ep2 in range(1):
                with autograd.record():
                        #eps = nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx) #
                        #eps2 = gluon.utils.split_and_load(nd.tanh(mu),ctx) #+nd.multiply(eps,sigma))#nd.random.uniform( low=-1, high=1, shape=fake_latent.shape, ctx=ctx)
			rec_output = [netDS(netDe(e)) for e in eps2]
                        fake_label = gluon.utils.split_and_load(nd.zeros(out_s_shape) ,ctx)
                        errGS = [GAN_loss(r, f) for r,f in zip(rec_output, fake_label)]
			for e in errGS:
                        	e.backward()
	        for idx, _ in enumerate(eps2):
			eps2[idx] = nd.tanh(eps2[idx]- lr/eps2[idx].shape[0]*eps2[idx].grad)
              images = netDe((eps2[0]))
              fake_img1T = nd.concat(images[0],images[1], images[2], dim=1)
              fake_img2T = nd.concat(images[3],images[4], images[5], dim=1)
              fake_img3T = nd.concat(images[6],images[7], images[8], dim=1)
              fake_img = nd.concat(fake_img1T,fake_img2T, fake_img3T,dim=2)
              visual.visualize(fake_img)
              plt.savefig('outputs/'+expname+str(ep2)+'_fakespost_'+str(epoch)+'.png')
	      #eps2 = nd.tanh(mu)#+nd.multiply(eps,sigma))#nd.random.uniform( low=-1, high=1, shape=fake_latent.shape, ctx=ctx)

            with autograd.record():
		#eps2 = gluon.utils.split_and_load(eps2,ctx)
                # Train with fake image
                # Use image pooling to utilize history imagesi
                output = [netD(f) for f in fake_concat]
                output2 = [netD2(f) for f in fake_latent]
                fake_label = nd.zeros(out_i_shape)
		fake_label = gluon.utils.split_and_load(fake_label, ctx)
                fake_latent_label = nd.zeros(out_l_shape )
		fake_latent_label = gluon.utils.split_and_load(fake_latent_label, ctx)
		eps =  gluon.utils.split_and_load( nd.random.uniform( low=-1, high=1, shape=latent_shape),ctx)
		rec_output = [netD(netDe(e)) for e in eps]
                errD_fake = [GAN_loss(r,f)  for r,f in zip(rec_output, fake_label)]
                errD_fake2 = [GAN_loss(o,f) for o,f in zip(output, fake_label )]
                errD2_fake = [GAN_loss(o, f) for o,f in zip(output2, fake_latent_label)]
                for f,o in zip(fake_label, rec_output):
			metric.update([f, ], [o, ])
		for f,o in zip(fake_latent_label,output2):
                	metric2.update([f, ], [o, ])
                real_concat =  nd.concat(real_in, real_out, dim=1) if append else  real_out
                output = [netD(r) for r in real_concat]
                output2 = [netD2(r) for r in real_latent]
                real_label = gluon.utils.split_and_load(nd.ones(out_i_shape ), ctx)
                real_latent_label = gluon.utils.split_and_load( nd.ones(out_l_shape ), ctx)
                errD_real = [GAN_loss(o, r) for o,r in zip (output, real_label )]
                errD2_real = [ GAN_loss(o, r) for o,r in zip (output2, real_latent_label )]
                for e1,e2,e4,e5 in zip (errD_real,errD_fake,errD2_real,errD2_fake):
			err = (e1 + e2) * 0.5+(e5 + e4) * 0.5
                	err.backward()
		for f,o in zip(real_label,output):
                	metric.update([f, ], [o, ])
                for f,o in zip(real_latent_label,output2):
            		metric2.update([f, ], [o, ])
            trainerD.step(batch.data[0].shape[0])
            trainerD2.step(batch.data[0].shape[0])
	    nd.waitall()  
	    with autograd.record():
		strong_output = [netDS(netDe(e)) for e in eps]
		strong_real = [netDS(f) for f in fake_concat]
		errs1 = [GAN_loss(r, f)  for r,f in zip(strong_output, fake_label)]
                errs2 = [GAN_loss(r, f) for r,f in zip(strong_real, real_label)]
		for f,s in zip ( fake_label, strong_output):
			metricStrong.update([f, ], [s, ])
                for f,s in zip ( real_label, strong_real):
			metricStrong.update([f, ], [s, ])
		for e1,e2 in zip (errs1,errs2):
                	strongerr = 0.5*(e1+e2)
			strongerr.backward()
	    trainerSD.step(batch.data[0].shape[0])
	    nd.waitall()
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
		sh = out_l_shape 
                #eps2 = nd.random_normal(loc=0, scale=1, shape=noiseshape, ctx=ctx) #
		#eps = nd.random.uniform( low=-1, high=1, shape=noiseshape, ctx=ctx)
		#if epoch>100:
                #        eps2 = nd.multiply(eps2,sigma)+mu
                #        eps2 = nd.tanh(eps2)
                #else:
                #eps = nd.random.uniform( low=-1, high=1, shape=noiseshape, ctx=ctx)
                #eps2 = nd.concat(eps,eps2,dim=0)
		rec_output = [netD(netDe(e)) for e in eps2]
                fake_latent= [(netEn(r)) for r in real_in]
                output2 = [netD2(f) for f in fake_latent]
                fake_out = [netDe(f) for f in fake_latent]
                fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
                output = [netD(f) for f in fake_concat]
                real_label =  gluon.utils.split_and_load(nd.ones(out_i_shape ), ctx)
                real_latent_label =  gluon.utils.split_and_load(nd.ones(out_l_shape ), ctx)
                errG2 = [GAN_loss(r, f) for r,f in zip(rec_output, real_label)]
                errR = [L1_loss(r, f) * lambda1 for r,f in zip(real_out, fake_out)]
                errG = [10*GAN_loss(r, f) for r,f in zip(output2, real_latent_label)] # +errG2+errR
		for e1,e2,e3  in  zip(errG,errG2,errR):
                	e = e1+e2+e3
			e.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
	    nd.waitall()  
            errD=(errD_real[0]+errD_fake[0])*0.5
            errD2=(errD2_real[0]+errD2_fake[0])*0.5
	    loss_rec_G2.append(nd.mean(errG2[0]).asscalar())
            loss_rec_G.append(nd.mean(nd.mean(errG[0])).asscalar()-nd.mean(errG2[0]).asscalar()-nd.mean(errR[0]).asscalar())
            loss_rec_D.append(nd.mean(errD[0]).asscalar())
            loss_rec_R.append(nd.mean(errR[0]).asscalar())
            loss_rec_D2.append(nd.mean(errD2[0]).asscalar())
            _, acc2 = metric2.get()
            name, acc = metric.get()
            acc_rec.append(acc)
            acc2_rec.append(acc2)

            # Print log infomation every ten batches
            if iter % 10 == 0:
                _, acc2 = metric2.get()
                name, acc = metric.get()
		_, accStrong = metricStrong.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                #print(errD)
		#logging.info('discriminator loss = %f, D2 loss = %f, generator loss = %f, G2 loss = %f, SD loss = %f,  D acc = %f , D2 acc = %f, DS acc = %f, reconstruction error= %f  at iter %d epoch %d'
                 #   	% (nd.mean(errD[0]).asscalar(),nd.mean(errD2[0]).asscalar(),
                 #     	nd.mean(errG[0]-errG2[0]-errR[0]).asscalar(),nd.mean(errG2[0]).asscalar(),nd.mean(strongerr[0]).asscalar() ,acc,acc2,accStrong[0],nd.mean(errR[0]).asscalar() ,iter, epoch))
                iter = iter + 1
        btic = time.time()
        name, acc = metric.get()
        _, acc2 = metric2.get()
        #tp_file = open(expname + "_trainloss.txt", "a")
        #tp_file.write(str(nd.mean(errG2).asscalar()) + " " + str(
        #    nd.mean(nd.mean(errG)).asscalar() - nd.mean(errG2).asscalar() - nd.mean(errR).asscalar()) + " " + str(
        #    nd.mean(errD).asscalar()) + " " + str(nd.mean(errD2).asscalar()) + " " + str(nd.mean(errR).asscalar()) +" "+str(acc) + " " + str(acc2)+"\n")
        #tp_file.close()
        metric.reset()
        metric2.reset()
        train_data.reset()
	metricStrong.reset()

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%2 ==0:# and epoch>0:
            text_file = open(expname + "_validtest.txt", "a")
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_parameters(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D2.params"
            netD2.save_parameters(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_parameters(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_parameters(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_SD.params"
            netDS.save_parameters(filename)
            fake_img1 = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img4 = nd.concat(real_in[3],real_out[3], fake_out[3], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                
                real_in = vbatch.data[0]
                real_out = vbatch.data[1]
                real_in  = gluon.utils.split_and_load(real_in, ctx)
            	real_out  = gluon.utils.split_and_load(real_out, ctx)

		fake_latent= [netEn(r) for r in real_in]
                fake_out = [netDe(f) for f in fake_latent]
                for f,r in zip(fake_out, real_out):
                	metricMSE.update([f, ], [r, ])
            _, acc2 = metricMSE.get()
	    toterrR = 0	
	    for e in errR:
		toterrR+=nd.mean(e).asscalar()
            text_file.write("%s %s %s\n" % (str(epoch), toterrR, str(acc2)))
            metricMSE.reset()
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

def mxctx():
    ct = []
    for i in range(100):
        try:
            mx.nd.zeros((1,), ctx=mx.gpu(i))
            ct.append(mx.gpu(i))
        except:
            return ct

def main(opt): 
    if opt.useAE == 1:
        useAE = True
    else:
        useAE = False
    if opt.seed != -1:
            random.seed(opt.seed)
    ctx = mxctx()  if opt.use_gpu else mx.cpu()
    inclasspaths , inclasses = dload.loadPaths(opt.dataset, opt.datapath, opt.expname, opt.batch_size+1, opt.classes)
    train_data, val_data = load_image.load_image(inclasspaths, opt.batch_size, opt.img_wd, opt.img_ht, opt.noisevar)
    print('Data loading done.')

    if opt.istest:
        print('testing not implemented')
    else:
	print("Latent variable dim : " + str(opt.latent))
        netEn, netDe,  netD, netD2, netDS, trainerEn, trainerDe, trainerD, trainerD2, trainerDS = set_network(opt.depth, ctx, opt.lr, opt.beta1,opt.ndf,  opt.ngf,opt.latent, opt.append)
        if opt.graphvis:
            print(netG)
        print('training')
        print(opt.epochs)
        loss_vec = train(int(opt.continueEpochFrom), opt.pool_size, opt.epochs, train_data,val_data, ctx, netEn, netDe,  netD, netD2,netDS,  trainerEn, trainerDe, trainerD, trainerD2,trainerDS, opt.lambda1, opt.batch_size, opt.expname,  opt.append, useAE = useAE)
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
