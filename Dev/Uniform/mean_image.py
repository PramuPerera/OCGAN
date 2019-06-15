import numpy as np
import dataloaderiter as dload
import mxnet as mx
def save_mean( dataset,datapath,expname,batch_size, classes, img_wd = 61, img_ht = 61):
    inclasspaths , inclasses = dload.loadPaths(dataset, datapath, expname, batch_size+1, classes)
    for  idx,img in enumerate(inclasspaths):
        img_arr = mx.image.imread(img).astype(np.float32)/127.5 - 1
        img_arr = mx.image.imresize(img_arr, img_wd, img_ht)
        img_arr = mx.nd.transpose(img_arr, (2, 0, 1))
	if idx == 0:
            mean_image = img_arr
        else:
            mean_image = (mean_image*float(idx)+ img_arr)/(float(idx)+1.0)
        
    np.save("mean.npy",mean_image.asnumpy())
    return mean_image

    
def load_mean( ):
    mean_image = np.load("mean.npy")
    return mx.nd.array(mean_image)


