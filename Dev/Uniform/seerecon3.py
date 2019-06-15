from random import shuffle
import options
import vaetest
import numpy as np
import os
import random
import mxnet as mx
import matplotlib.pyplot as plt
import mxnet.ndarray as nd
import visual
import dataloaderiter as dload
from skimage import exposure
#random.seed(1000)
opt = options.test_options()
opt.istest = 0

def heq(image, nbins ):
    image = (image+1)*0.5
    cdf, bin_centers = exposure.cumulative_distribution(image, nbins)
    cdf = np.insert(cdf,0,0)
    cdf = np.append(cdf,1)
    bin_centers = np.insert(bin_centers,0,0)
    bin_centers = np.append(bin_centers,1)
    print(bin_centers)
    print(cdf)
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



#First read all classes one at a time and iterate through all
#text_file = open(opt.dataset + "_folderlist.txt", "r")
#folders = text_file.readlines()
#text_file.close()
#folders = [i.split('\n', 1)[0] for i in folders]
follist = range(0,201,10)
folders = range(0,10)
for classname in [8]: #folders:


    	ctx = mx.gpu() if opt.use_gpu else mx.cpu()
	testclasspaths = []
	testclasslabels = []
	with open(opt.dataset+"_"+opt.expname+'_testlist.txt' , 'r') as f:
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

    	testclasslabels, testclasspaths = zip(*c)
    	testclasslabels = testclasslabels[1:5000]
    	testclasspaths = testclasspaths[1:5000]

	

    	netEn,netDe, netD, netD2 = vaetest.set_network(opt.depth, ctx, 0, 0, opt.ndf, opt.ngf, opt.append)
    	netEn.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_En.params', ctx=ctx)
    	netDe.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_De.params', ctx=ctx)
    	netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
    	#netD2.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D2.params', ctx=ctx)

	#fakecode = nd.random.uniform(low = -1, high = 1, shape=(16, 4096,1,1), ctx=ctx)
        fakecode = nd.random.uniform(low = -1, high = 1, shape=(16, 64,1,1), ctx=ctx)
	out = netDe(fakecode)
	import load_image
	test_data = load_image.load_test_images(testclasspaths,testclasslabels,opt.batch_size, opt.img_wd, opt.img_ht, ctx, opt.noisevar)
    	for batch in (test_data):
        	real_in = batch.data[0].as_in_context(ctx)
        code = netEn(real_in)
	codecopy = code.copy()
	eq_code = heq(code.asnumpy(), nbins=2)
        code = nd.array(eq_code, ctx=ctx)
	recon = netDe(code)
	
	print(nd.max(netEn(real_in)))
        print(nd.min(netEn(real_in)))
	print(real_in.shape)
	print(recon.shape)
	fake_img1 = nd.concat(real_in[0],recon[0], recon[0], out[3],dim=1)
	fake_img2 = nd.concat(out[7],out[6], out[5], out[4],dim=1)
	fake_img3 = nd.concat(out[8],out[9], out[10], out[11],dim=1)
	fake_img4 = nd.concat(out[15],out[14], out[13], out[12],dim=1)        
	fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
        #print(np.shape(fake_img))
        #visual.visualize(fake_img)
	tep = code[0].asnumpy()
	tep.flatten()
	plt.figure(figsize=(20,10))
	print(np.shape(code[0].asnumpy().flatten()))
 	dec = netDe(code)
	dec2 = netDe(fakecode)	
	plt.subplot(6,4,1)
	plt.hist((codecopy[0].asnumpy().flatten() ),100)
        plt.subplot(6,4,2)
        plt.hist((code[0].asnumpy().flatten() ),100)
        plt.subplot(6,4,3)
        plt.imshow((((real_in[0]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,4,4)
	plt.imshow((((dec[0]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        
	plt.subplot(6,4,5)
        plt.hist((codecopy[1].asnumpy().flatten() ),100)
	plt.subplot(6,4,6)
        plt.hist((code[1].asnumpy().flatten() ),100)
        plt.subplot(6,4,7)
        plt.imshow((((real_in[1]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,4,8)
        plt.imshow((((dec[1]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        
        plt.subplot(6,4,9)
        plt.hist((codecopy[2].asnumpy().flatten() ),100)
	plt.subplot(6,4,10)
        plt.hist((code[2].asnumpy().flatten() ),100)
        plt.subplot(6,4,11)
        plt.imshow((((real_in[2]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,4,12)
        plt.imshow((((dec[2]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))

        plt.subplot(6,4,13)
        plt.hist((codecopy[3].asnumpy().flatten() ),100)
	plt.subplot(6,4,14)
 	plt.hist((code[3].asnumpy().flatten()),100)
        plt.subplot(6,4,15)
        plt.imshow((((real_in[3]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,4,16)
        plt.imshow((((dec[3]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))

        plt.subplot(6,4,17)
        plt.hist((codecopy[4].asnumpy().flatten() ),100)
        plt.subplot(6,4,18)
        plt.hist((code[4].asnumpy().flatten()),100)
        plt.subplot(6,4,19)
        plt.imshow((((real_in[4]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,4,20)
        plt.imshow((((dec[4]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))

        plt.subplot(6,4,21)
        plt.hist((codecopy[5].asnumpy().flatten() ),100)
	plt.subplot(6,4,22)
        plt.hist((code[5].asnumpy().flatten()),100)
        plt.subplot(6,4,23)
        plt.imshow((((real_in[5]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,4,24)
        plt.imshow((((dec[5]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.savefig('outputs/dist_'+opt.expname+'_.png')
