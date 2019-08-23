import os
import numpy as np


def loadPaths(opt):
    dataset = opt.dataset
    datapath = opt.datapath
    classes = opt.classes
    expname = opt.expname
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
        
    # read names of training classes
    text_file = open(dataset + "_trainlist.txt", "r")
    folders = text_file.readlines()
    text_file.close()
    folders = [i.split('\n', 1)[0] for i in folders]
    inclasspaths = []
    testclasspaths = []
    inclasslabels = []
    testclasslabels = []
    # if classes is set to a a value use it instead
    inclasses = list(folders)
    if classes != "":
        inclasses = [classes]
    print(inclasses)
    if opt.protocol == 1:
        # first 80% of each image is treated as training. remainder is treated as testing
        for lbl, nclass in enumerate(inclasses):
            dirs = os.listdir(datapath + dataset + '/' + nclass)
            for nfile in range(int(len(dirs)*0.8)):
                inclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
                inclasslabels.append(lbl)
            for nfile in range(int(len(dirs)*0.8)+1, len(dirs)):
                testclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
                testclasslabels.append(lbl)
        text_file = open(dataset + "_novellist.txt", "r")
        folders = text_file.readlines()
        text_file.close()
        folders = [i.split('\n', 1)[0] for i in folders]
        print(folders)
        cluttersize = int(round(len(testclasslabels)/len(folders)))
        for i in range(len(folders) ):
            dirs = os.listdir(datapath + dataset + '/' + folders[i])
            for nfile in dirs[0: cluttersize]:
                testclasspaths.append(datapath + dataset + '/' +folders[i] + '/' + nfile)
                testclasslabels.append(-1)
        # write test files and labels to external file for future testing
        text_file = open(dataset + "_" + expname + "_testlist.txt", "w")
        for fn, lbl in zip(testclasspaths, testclasslabels):
            text_file.write("%s %s\n" % (fn, str(lbl)))
        text_file.close()

        text_file = open(dataset + "_" + expname + "_trainlist.txt", "w")
        for fn, lbl in zip(inclasspaths, inclasslabels):
            text_file.write("%s %s\n" % (fn, str(lbl)))
        text_file.close()
        

    else:
        # Use train / test split
        for lbl, nclass in enumerate(inclasses):
            dirs = os.listdir(datapath + dataset + '/training/' + nclass)
            for nfile in range(int(len(dirs))):
                inclasspaths.append(datapath + dataset + '/training/' + nclass + '/' + dirs[nfile])
                inclasslabels.append(lbl)

        for lbl, nclass in enumerate(inclasses):
            dirs = os.listdir(datapath + dataset + '/testing/' + nclass)
            for nfile in range(int( len(dirs))):
                testclasspaths.append(datapath + dataset + '/testing/' + nclass + '/' + dirs[nfile])
                testclasslabels.append(lbl)
        validationclasspaths = list(testclasspaths)
        validationclasslabels = list(testclasslabels)
        text_file = open(dataset + "_novellist.txt", "r")
        folders = text_file.readlines()
        text_file.close()
        folders = [i.split('\n', 1)[0] for i in folders]
        cluttersize = int(round(len(testclasslabels)/len(folders)))
        for i in range(len(folders) ):
            dirs = os.listdir(datapath + dataset + '/testing/' + folders[i])
            for nfile in dirs:
                    testclasspaths.append(datapath + dataset + '/testing/' +folders[i] + '/' + nfile)
                    testclasslabels.append(-1)
        # write test files and labels to external file for future testing
        text_file = open(dataset + "_" + expname + "_testlist.txt", "w")
        for fn, lbl in zip(testclasspaths, testclasslabels):
            text_file.write("%s %s\n" % (fn, str(lbl)))
        text_file.close()
        #consider 'other' classes and get paths of their samples for validation

        text_file = open(dataset + "_" + expname + "_trainlist.txt", "w")
        for fn, lbl in zip(inclasspaths, inclasslabels):
            text_file.write("%s %s\n" % (fn, str(lbl)))
        text_file.close()
    return [inclasspaths, inclasslabels]

