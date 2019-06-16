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
	pert = netDe(code+nd.random.normal(loc=0, scale=0.2, shape=code.shape,ctx=ctx))	
	pert2 = netDe(code+nd.random.normal(loc=0, scale=0.2, shape=code.shape,ctx=ctx))
	print(nd.max(netEn(real_in)))
        print(nd.min(netEn(real_in)))
	print(real_in.shape)
	print(recon.shape)
	fake_img0 = nd.concat(real_in[0],recon[0], pert[0],pert2[0],dim=1)
	fake_img1 = nd.concat(real_in[1],recon[1], pert[1],pert2[1],dim=1)
	fake_img2 = nd.concat(real_in[2],recon[2], pert[2],pert2[2],dim=1)
	fake_img3 = nd.concat(real_in[3],recon[3], pert[3],pert2[3],dim=1)
	fake_img4 = nd.concat(real_in[4],recon[4], pert[4],pert2[4],dim=1)
	fake_img5 = nd.concat(real_in[5],recon[5], pert[5],pert2[5],dim=1)


	fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4,fake_img5,fake_img0, dim=2)
        #print(np.shape(fake_img))
        visual.visualize(fake_img)
	plt.savefig('outputs/dist_'+opt.expname+'_.png')
