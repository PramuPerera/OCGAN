import mxnet as mx
from random import shuffle
from mxnet import gluon
from mxnet import ndarray as nd
import numpy as np

def load_image(fnames, batch_size, img_wd, img_ht,noisevar=0.2 , is_reversed=False):
    img_in_list = []
    img_out_list = []
    shuffle(fnames)
    print(np.shape(fnames))
    for img in fnames:
        img_arr = mx.image.imread(img).astype(np.float32)/127.5 - 1
	#print(nd.max(img_arr[0]))
        img_arr = mx.image.imresize(img_arr, img_wd, img_ht)
        # Crop input and output images
        croppedimg = mx.image.fixed_crop(img_arr, 0, 0, img_wd, img_ht)
	if noisevar>0: 
        	img_arr_in, img_arr_out = [croppedimg+mx.random.normal(0, noisevar, croppedimg.shape),
                	                   croppedimg]
	else: 
		img_arr_in, img_arr_out = [croppedimg, croppedimg]
        img_arr_in, img_arr_out = [nd.transpose(img_arr_in, (2, 0, 1)),
                                   nd.transpose(img_arr_out, (2, 0, 1))]
        img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                   img_arr_out.reshape((1,) + img_arr_out.shape)]
        img_in_list.append(img_arr_out if is_reversed else img_arr_in)
        img_out_list.append(img_arr_in if is_reversed else img_arr_out)

    train_list_in = img_in_list[0:int(len(img_in_list)*0.9)]
    train_list_out =  img_out_list[0:int(len(img_out_list)*0.9)]
    val_list_in = img_in_list[int(len(img_in_list)*0.9):-1]
    val_list_out = img_out_list[int(len(img_out_list)*0.9):-1]

    itertrain = mx.io.NDArrayIter(data=[nd.concat(*train_list_in, dim=0), nd.concat(*train_list_out, dim=0)],
                                  batch_size=batch_size)
    iterval = mx.io.NDArrayIter(data=[nd.concat(*val_list_in, dim=0), nd.concat(*val_list_out, dim=0)],
                                batch_size=int(batch_size/5.0))

    return [itertrain, iterval]



def load_test_images(fnames, lbl, batch_size, img_wd, img_ht, ctx, noisevar=0.2, is_reversed=False):
    img_in_list = []
    img_out_list = []
    #shuffle(fnames)
    for img in fnames:
        img_arr = mx.image.imread(img).astype(np.float32)/127.5 - 1
        img_arr = mx.image.imresize(img_arr, img_wd, img_ht)
        # Crop input and output images
        croppedimg = mx.image.fixed_crop(img_arr, 0, 0, img_wd, img_ht)
	if noisevar>0:
       		img_arr_in, img_arr_out = [croppedimg+mx.random.normal(0, noisevar , croppedimg.shape),
                	                   croppedimg]
        else:
                img_arr_in, img_arr_out = [croppedimg, croppedimg]
	img_arr_in, img_arr_out = [nd.transpose(img_arr_in, (2, 0, 1)),
                                   nd.transpose(img_arr_out, (2, 0, 1))]
        img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                   img_arr_out.reshape((1,) + img_arr_out.shape)]
        img_in_list.append(img_arr_out if is_reversed else img_arr_in)
        img_out_list.append(img_arr_in if is_reversed else img_arr_out)

    tempdata = [nd.concat(*img_in_list, dim=0), nd.concat(* img_out_list, dim=0)]
    templbl = mx.nd.array(lbl)
    itertest = mx.io.NDArrayIter(data=tempdata, label=templbl,
                                  batch_size=batch_size)
    return itertest
