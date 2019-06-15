from __future__ import print_function
import math
import seaborn as sns
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
import random
import argparse
from skimage import exposure
import mean_image
#logging.basicConfig()


def heq(image, nbins ):
    image = (image+1)*0.5
    cdf, bin_centers = exposure.cumulative_distribution(image, nbins)
    #cdf = np.insert(cdf,0,0)
    #cdf = np.append(cdf,1)
    #bin_centers = np.insert(bin_centers,0,0)
    #bin_centers = np.append(bin_centers,1)
    out = np.interp(image.flat, bin_centers, cdf)
    out = (2*out)-1
    return out.reshape(image.shape)



def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()



def set_network(depth, ctx, lr, beta1, ndf,ngf, append=True):
    if append:
            netD = models.Discriminator(in_channels=6, n_layers=depth-1, istest=True, ndf=ndf)
    else:        
            netD = models.Discriminator(in_channels=3, n_layers=depth-1, istest=True, ndf=ndf)
    netEn = models.Encoder(in_channels=3, n_layers=depth, istest=True, latent=4096,ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netDe = models.Decoder(in_channels=3, n_layers=depth, istest=True, latent=4096,ndf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netD2 = models.LatentDiscriminator(in_channels=6, n_layers =2 , ndf=ndf)
    netDS = models.Discriminator(in_channels=3, n_layers =2 , ndf=64)

    return netEn, netDe,  netD, netD2 ,netDS

def main(opt):
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    testclasspaths = []
    testclasslabels = []
    print('loading test files')
    filename = '_testlist.txt'
    with open(opt.dataset+"_"+opt.expname+filename , 'r') as f:
        for line in f:
            testclasspaths.append(line.split(' ')[0])
            if int(line.split(' ')[1])==-1:
                testclasslabels.append(0)
            else:
                testclasslabels.append(1)
    neworder = range(len(testclasslabels))
    neworder = shuffle(neworder)
    
    c = list(zip(testclasslabels, testclasspaths))
    print('shuffling')
    random.shuffle(c)
    im_mean = mean_image.load_mean( )
    im_mean = im_mean.broadcast_to((opt.batch_size,np.shape(im_mean)[0], np.shape(im_mean)[1],np.shape(im_mean)[2]))#im_mean = nd.transpose(im_mean, (2, 0, 1))#testclasslabels, testclasspaths = zip(*c)
    #testclasslabels = testclasslabels[1:5000]
    #testclasspaths = testclasspaths[1:5000]
    ltnt = 4096
    print('loading pictures')
    test_data = load_image.load_test_images(testclasspaths,testclasslabels,opt.batch_size, opt.img_wd, opt.img_ht, ctx, opt.noisevar)
    print('picture loading done')
    netEn,netDe, netD, netD2 , netDS= set_network(opt.depth, ctx, 0, 0, opt.ndf, opt.ngf, opt.append)
    netEn.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_En.params', ctx=ctx)
    netDe.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_De.params', ctx=ctx)
    netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
    netD2.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D2.params', ctx=ctx)
    netDS.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_SD.params', ctx=ctx)
    print('Model loading done')
    lbllist = [];
    scorelist1 = [];
    scorelist2 = [];
    scorelist3 = [];
    scorelist4 = [];
    test_data.reset()
    count = 0
    for batch in (test_data):
        count+=1
	print (str(count)) #, end="\r")
        real_in = batch.data[0].as_in_context(ctx)-im_mean.as_in_context(ctx)
        real_out = batch.data[1].as_in_context(ctx)-im_mean.as_in_context(ctx)
        lbls = batch.label[0].as_in_context(ctx)
	code = netEn((real_out))
	code=code+nd.random.normal(loc=0, scale=0.002, shape=code.shape,ctx=ctx)
        outnn = (netDe(code))
        out_concat =  nd.concat(real_out, outnn, dim=1) if opt.append else  outnn
        output4 = nd.mean((netD(out_concat)), (1, 3, 2)).asnumpy()    
	code = netEn(real_in)
	#code=codet+nd.random.normal(loc=0, scale=0.0000001, shape=code.shape,ctx=ctx)
	#code2=codet+nd.random.normal(loc=0, scale=0.000001, shape=code.shape,ctx=ctx)
	#eq_code = heq(code.asnumpy(),2)
	#code = nd.array(eq_code, ctx=ctx)
        out = netDe(code)
	#out2 = netDe(code2)
        out_concat =  nd.concat(real_in, out, dim=1) if opt.append else  out
        output = netD(out_concat) #Denoised image
        output3 = nd.mean((out-real_out)**2, (1, 3, 2)).asnumpy() #denoised-real
        output = nd.mean(output, (1, 3, 2)).asnumpy()
        out_concat =  nd.concat(real_out, real_out, dim=1) if opt. append else  real_out
	
        output2 = netDS(netDe(code)) #Image with no noise
        output2 = nd.mean(output2, (1,3,2)).asnumpy()
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
    if not opt.isvalidation:

	
	    fpr, tpr, _ = roc_curve(lbllist, scorelist1, 1)
	    roc_auc1 = auc(fpr, tpr)
	    fpr, tpr, _ = roc_curve(lbllist, scorelist2, 1)
	    roc_auc2 = auc(fpr, tpr)
	    fpr, tpr, _ = roc_curve(lbllist, scorelist3, 1)
	    roc_auc3 = auc(fpr, tpr)
	    EER = fpr[np.nanargmin(np.absolute((1-tpr - fpr)))]
	    fpr, tpr, _ = roc_curve(lbllist, scorelist4, 1)
	    roc_auc4 = auc(fpr, tpr)
	    plt.gcf().clear()
            plt.clf()
            sns.set(color_codes=True)
            posscores =  [scorelist3[i]  for i , v  in enumerate(lbllist)  if  int(v)==1]
            negscores = [scorelist3[i]  for i , v  in enumerate(lbllist)  if  int(v)==0]
            #sns.distplot(posscores, hist=False, label="Known Classes" ,rug=True)
            sns.kdeplot(posscores, label="Known Classes" )
            sns.kdeplot(negscores, label="Unnown Classes" )
            #plt.hold()
            #sns.distplot(negscores, hist=False, label = "Unknown Classes", rug=True);
            plt.legend()
            plt.savefig('outputs/matdist_'+opt.expname+'_.png')


            plt.gcf().clear()
            inputT = nd.zeros((ltnt,ltnt,1,1),ctx=ctx)
            for i in range(0,ltnt):
                inputT[i,i,:,:] = -1
            out = netDe(inputT)
            count = 0
            for i in range(int(math.ceil(math.sqrt(ltnt)))):
                for j in range(int(math.ceil(math.sqrt(ltnt)))):
                   if count<ltnt:
                        plt.subplot(math.ceil(math.sqrt(ltnt)),math.ceil(math.sqrt(ltnt)),count+1)
                        plt.imshow(((out[count].asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
                        plt.axis('off')
                   count+=1
            plt.savefig('outputs/atoms_'+opt.expname+'_.png')
	    plt.gcf().clear()
            plt.clf()
	    print(EER)
	    return([roc_auc1, roc_auc2, roc_auc3, roc_auc4])
    else:
	    return([0,0,0,0])
    fakecode = nd.random_normal(loc=0, scale=1, shape=(16, 4096,1,1), ctx=ctx)
    out = netDe(fakecode)
    fake_img1 = nd.concat(out[0],out[1], out[2], out[3],dim=1)
    fake_img2 = nd.concat(out[7],out[6], out[5], out[4],dim=1)
    fake_img3 = nd.concat(out[8],out[9], out[10], out[11],dim=1)
    fake_img4 = nd.concat(out[15],out[14], out[13], out[12],dim=1)
    fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
    #print(np.shape(fake_img))
    visual.visualize(fake_img)
    plt.savefig('outputs/fakes_'+opt.expname+'_.png')

if __name__ == "__main__":
    opt = options.test_options()
    roc_auc = main(opt)
    print(roc_auc)
