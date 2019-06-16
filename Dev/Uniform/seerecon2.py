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
#random.seed(1000)
opt = options.test_options()
opt.istest = 0

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
	plt.subplot(6,2,1)
	plt.hist((code[0].asnumpy().flatten() ),100)
        plt.subplot(6,2,2)
	plt.imshow((((dec[0]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,2,3)
        plt.hist((code[1].asnumpy().flatten() ),100)
        plt.subplot(6,2,4)
        plt.imshow((((dec[1]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,2,5)
        plt.hist((code[2].asnumpy().flatten() ),100)
        plt.subplot(6,2,6)
        plt.imshow((((dec[2]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.subplot(6,2,7)
 	plt.hist((fakecode[0].asnumpy().flatten()),100)
        plt.subplot(6,2,8)
        plt.imshow((((dec2[0]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
        plt.subplot(6,2,9)
        plt.hist((fakecode[1].asnumpy().flatten()),100)
        plt.subplot(6,2,10)
        plt.imshow((((dec2[1]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.subplot(6,2,11)
        plt.hist((fakecode[0].asnumpy().flatten()),100)
        plt.subplot(6,2,12)
        plt.imshow((((dec2[2]).asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
	plt.savefig('outputs/dist_'+opt.expname+'_.png')
