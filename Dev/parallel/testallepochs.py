import options
import vaetest
import numpy as np
import os
import random
random.seed(1000)
opt = options.test_options()
opt.istest = 0

text_file = open(opt.dataset + "_progressEpoch.txt", "w")
text_file.close()
#First read all classes one at a time and iterate through all
#text_file = open(opt.dataset + "_folderlist.txt", "r")
#folders = text_file.readlines()
#text_file.close()
#folders = [i.split('\n', 1)[0] for i in folders]

follist = range(0,201,10)
folders = range(0,10)
for epoch in follist:
        opt.epochs = epoch
	roc_auctrain = vaetest.main(opt)
    	text_file = open(opt.dataset + "_progressEpoch.txt", "a")
        text_file.write("%s %s %s %s %s\n" % (str(epoch), str(roc_auctrain[0]), str(roc_auctrain[1]),str(roc_auctrain[2]),str(roc_auctrain[3]  )))
        text_file.close()
