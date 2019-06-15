import os
import numpy as np
def loadPaths(dataset, datapath, expname, minquery = 16, classes=""):
    # read names of classes; treat last class as clutte
    print("Data load")
    print(dataset + "_folderlist.txt")
    text_file = open(dataset + "_folderlist.txt", "r")
    folders = text_file.readlines()
    text_file.close()
    folders = [i.split('\n', 1)[0] for i in folders]
    valid_folders = []
    inclasspaths = []
    testclasspaths = []
    inclasslabels = []
    testclasslabels = []
    print(folders)
    # randomly pick 3 classes making sure each has more than 150 images
    for i in range(len(folders) - 1):
        dirs = os.listdir(datapath + dataset + '/' + folders[i])
        if len(dirs) > 150+minquery:
            valid_folders.append(folders[i])
    inclasses = np.random.permutation(np.arange(len(valid_folders)))[0]#[0:3]
    inclasses = [inclasses]
    inclasses = [valid_folders[i] for i in inclasses]
    #inclasses = ['092.grapes', '109.hot-tub', '148.mussels']
    #inclasses = ['092.grapes']
    #inclasses = [inclasses]
    #print(inclasses)
    if classes != "":
	inclasses = [classes]
    print(inclasses)
    # first 150 of each image is treated as training. remainder is treated as testing
    for lbl, nclass in enumerate(inclasses):
        dirs = os.listdir(datapath + dataset + '/' + nclass)
        for nfile in range(int(len(dirs)*0.8)):
            inclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
            inclasslabels.append(lbl)
        for nfile in range(int(len(dirs)*0.8)+1, len(dirs)):
            testclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
            testclasslabels.append(lbl)
    validationclasspaths = list(testclasspaths)
    validationclasslabels = list(testclasslabels)
    # pick 50% of images from clutter class
    cluttersize = int(round(len(testclasslabels)))
    dirs = os.listdir(datapath + dataset + '/' + folders[-1])
    for nfile in range(min(cluttersize, len(dirs))):
        testclasspaths.append(datapath + dataset + '/' + folders[-1] + '/' + dirs[nfile])
        testclasslabels.append(-1)
    # write test files and labels to external file for future testing
    text_file = open(dataset + "_" + expname + "_testlist.txt", "w")
    for fn, lbl in zip(testclasspaths, testclasslabels):
        text_file.write("%s %s\n" % (fn, str(lbl)))
    text_file.close()
    #consider 'other' classes and get paths of their samples for validation
    validation_set = list(set(folders)^set(inclasses))
    print(validation_set)
    nofentriesperclass = 1+int(np.size(inclasslabels)/np.size(validation_set))
    print(nofentriesperclass)
    for nclass in validation_set:
         dirs = os.listdir(datapath + dataset + '/' + nclass)
         for nfile in range(min(len(dirs),nofentriesperclass)):
            validationclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
            validationclasslabels.append(-1)
    text_file = open(dataset + "_" + expname + "_validationlist.txt", "w")        
    for fn, lbl in zip(validationclasspaths, validationclasslabels):
        text_file.write("%s %s\n" % (fn, str(lbl)))
    text_file.close()

    text_file = open(dataset + "_" + expname + "_trainlist.txt", "w")
    for fn, lbl in zip(inclasspaths, inclasslabels):
        text_file.write("%s %s\n" % (fn, str(lbl)))
    text_file.close()    
    return [inclasspaths, inclasslabels]

