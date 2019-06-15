import options
import TestNovelty
import numpy as np
import os
import random
from shutil import copyfile
random.seed(1000)
opt = options.test_options()
opt.istest = 0
text_file = open(opt.dataset + "_progress.txt", "w")
text_file.close()
# First read all classes one at a time and iterate through all
text_file = open(opt.dataset + "_folderlist.txt", "r")
folders = text_file.readlines()
text_file.close()
folders = [i.split('\n', 1)[0] for i in folders]
epoch_original = opt.epochs
for classname in folders:
    filelisttext = open(opt.dataset+'_trainlist.txt', 'w')
    filelisttext.write(str(classname))
    filelisttext.close()
    filelisttext = open(opt.dataset+'_novellist.txt','w')
    novellist = list(set(folders)-set([classname]))
    opt.epochs = epoch_original
    for novel in novellist:
        filelisttext.write(str(novel)+'\n')
    filelisttext.close()
    epoch = []
    trainerr = []
    valerr =[]
    trainstring = "python2 TrainNovelty.py --epochs " + str(opt.epochs) + ' --batch_size ' + str(opt.batch_size) +' --ndf ' +  str(opt.ndf) +' --ngf ' + str(opt.ngf) +' --istest 0 --expname ' + opt.expname +' --img_wd ' + str(opt.img_wd) + ' --img_ht ' + str(opt.img_ht)+ ' --depth ' + str(opt.depth)+ ' --datapath ' + opt.datapath + ' --noisevar ' + str(opt.noisevar) +' --lambda1 ' + str(500)+ ' --seed 1000  --append 0  --dataset ' + opt.dataset + ' --ntype ' +str( opt.ntype) +' --latent '+str(opt.latent)
    os.system(trainstring)
    res_file = open(opt.expname + "_validtest.txt", "r")
    results = res_file.readlines()
    res_file.close()
    results = [i.split('\n', 1)[0] for i in results]
    for line in results:
            temp = line.split(' ', 1)
            epoch.append(temp[0])
            temp = temp[1].split(' ', 1)
            trainerr.append(temp[0])
            valerr.append(temp[1])
    valep = np.argmin(np.array(valerr))
    trainep = np.argmin(np.array(trainerr))
    opt.epochs = epoch[valep]
    roc_aucval = TestNovelty.main(opt)
    opt.epochs = epoch[trainep]
    roc_auctrain = TestNovelty.main(opt)
    text_file = open(opt.dataset + "_progress.txt", "a")
    text_file.write("%s %s %s %s %s %s %s %s %s %s\n" % (str(valerr[valep]), str(trainerr[trainep]), str(roc_aucval[0]),str(roc_auctrain[0]), str(roc_aucval[1]),str(roc_auctrain[1]), str(roc_aucval[2]),str(roc_auctrain[2]), str(roc_aucval[3]),str(roc_auctrain[3]  )))
    text_file.close()
    copyfile('checkpoints/'+opt.expname+"_"+str(follist[valep])+"_En.params", 'checkpoints/'+opt.expname+"_class"+str(classname)+"_"+str(follist[valep])+"_En.params")
    copyfile('checkpoints/'+opt.expname+"_"+str(follist[valep])+"_De.params", 'checkpoints/'+opt.expname+"_class"+str(classname)+"_"+str(follist[valep])+"_De.params")

