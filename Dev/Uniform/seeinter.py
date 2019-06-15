import options
import vaetest
import numpy as np
import os
import random
import mxnet as mx
import matplotlib.pyplot as plt
import mxnet.ndarray as nd
import visual
import load_image
random.seed(1000)
opt = options.test_options()
opt.istest = 0

text_file = open(opt.dataset + "_progress.txt", "w")
text_file.close()
#First read all classes one at a time and iterate through all
#text_file = open(opt.dataset + "_folderlist.txt", "r")
#folders = text_file.readlines()
#text_file.close()
#folders = [i.split('\n', 1)[0] for i in folders]

ctx = mx.gpu() if opt.use_gpu else mx.cpu()

testclasspaths = []
testclasslabels = []
filename = '_trainlist.txt'
with open(opt.dataset+"_"+opt.expname+filename , 'r') as f:
        for line in f:
            testclasspaths.append(line.split(' ')[0])
            if int(line.split(' ')[1])==-1:
                testclasslabels.append(0)
            else:
                testclasslabels.append(1)
#testclasspaths = testclasspaths[0:1]
#testclasslabels = testclasslabels[0:1]
test_data = load_image.load_test_images(testclasspaths,testclasslabels,opt.batch_size, opt.img_wd, opt.img_ht, ctx, opt.noisevar)

print(ctx)





follist = range(0,201,10)
folders = range(0,10)
for classname in [0]: #folders:

    	#ctx = mx.gpu() if opt.use_gpu else mx.cpu()
	
	

    	netEn,netDe, netD, netD2 = vaetest.set_network(opt.depth, ctx, 0, 0, opt.ndf, opt.ngf, opt.append)
    	netEn.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_En.params', ctx=ctx)
    	netDe.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_De.params', ctx=ctx)
    	netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
    	netD2.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D2.params', ctx=ctx)
	print(ctx)
	test_data = test_data.next()
	print(np.shape(test_data.data))
	imc1 = netEn(test_data.data[0][0].expand_dims(1).as_in_context(ctx))
	im1 = netDe(imc1)
        imc2 = netEn(test_data.data[0][1].expand_dims(1).as_in_context(ctx))
        im2 = netDe(imc2)
	print(np.shape(im1))
	fakecode = nd.random_normal(loc=0, scale=1, shape=(16, 32,1,1), ctx=ctx)
	out = netDe(fakecode)
        fake_img1 = nd.concat(im1,out[1], out[2], out[3],dim=1)
	fake_img2 = nd.concat(out[7],out[6], out[5], out[4],dim=1)
	fake_img3 = nd.concat(out[8],out[9], out[10], out[11],dim=1)
	fake_img4 = nd.concat(out[15],out[14], out[13], im2 ,dim=1)        
	fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
        #print(np.shape(fake_img))
        visual.visualize(fake_img)
        plt.savefig('outputs/fakes_'+opt.expname+'_.png')
