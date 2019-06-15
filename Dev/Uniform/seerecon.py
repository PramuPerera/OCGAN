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
rd = random.random()*1000
print(rd)
random.seed(int(rd))
opt = options.test_options()
opt.istest = 0
mx.random.seed(int(rd))
#First read all classes one at a time and iterate through all
#text_file = open(opt.dataset + "_folderlist.txt", "r")
#folders = text_file.readlines()
#text_file.close()
#folders = [i.split('\n', 1)[0] for i in folders]
follist = range(0,201,10)
folders = range(0,10)
for classname in [8]: #folders:


    	ctx = mx.gpu() if opt.use_gpu else mx.cpu()



    	netEn,netDe, netD, netD2 = vaetest.set_network(opt.depth, ctx, 0, 0, opt.ndf, opt.ngf, opt.append)
    	netEn.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_En.params', ctx=ctx)
    	netDe.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_De.params', ctx=ctx)
    	netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
    	#netD2.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D2.params', ctx=ctx)
	fakecode = nd.random.uniform(low = -1, high = 1, shape=(16, 64,1,1), ctx=ctx)
	#fakecode = 0.5*nd.normal(0.7, 1, shape=(16, 128,1,1), ctx=ctx)+0.5*nd.normal(0.5, 1, shape=(16, 128,1,1), ctx=ctx)
	#fakecode = nd.normal(-0.7, 1, shape=(16, 128,1,1), ctx=ctx)
	out = netDe(fakecode)
        fake_img1 = nd.concat(out[0],out[1], out[2], out[3],dim=1)
	fake_img2 = nd.concat(out[7],out[6], out[5], out[4],dim=1)
	fake_img3 = nd.concat(out[8],out[9], out[10], out[11],dim=1)
	fake_img4 = nd.concat(out[15],out[14], out[13], out[12],dim=1)        
	fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
        #print(np.shape(fake_img))
	fakecode = nd.random.uniform(low = -1, high = 1, shape=(16, 64,1,1), ctx=ctx)
	aakecode = nd.random.uniform(low = -1, high = 1, shape=(16, 64,1,1), ctx=ctx)
        visual.visualize(fake_img)
        plt.savefig('outputs/fakes_'+opt.expname+'_.png')
