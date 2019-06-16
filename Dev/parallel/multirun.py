import options
import ocgantestdisjoint
import cvpr
import numpy as np
import random
import os
opt = options.test_options()

opt.istest = 0
#First use the validation set to pick best model:wq
text_file = open(opt.dataset + "_progress.txt", "w")
text_file.close()
text_file1 = open(opt.dataset + "_progress1.txt", "w")
text_file1.close()
for itt in range(50):
        text_file1 = open(opt.dataset + "_progress1.txt", "a")        
	text_file = open(opt.dataset + "_progress.txt", "a")
	os.system('python2  cvprappend.py --epochs 1001 --ndf 16 --ngf 64  --expname grapesip64 --img_wd 125  --img_ht 125  --depth 3  --datapath ../ --noisevar 0.2  --lambda1 500 --seed 1000 --append 0')

	auc1 = []
	auc2=[]
	auc3=[]
	auc4=[]
	ran = range(0,1000,10)
	for i in ran:
	    opt.epochs = i
	    istest = 0
	    roc_auc = ocgantestdisjoint.main(opt)
	    print(roc_auc)
	    auc1.append(roc_auc[0])
	    auc2.append(roc_auc[1])
	    auc3.append(roc_auc[2])
	    auc4.append(roc_auc[3])

	#Pick best model w.r.t criterion 1

        imax = np.argmax(np.array(auc1))
	opt.epochs = ran[imax]
	opt.istest=1
        res1 = ocgantestdisjoint.main(opt)[0]
        print(res1)
        
        imax = np.argmax(np.array(auc2))
        opt.epochs = ran[imax]
        opt.istest=1
        res2 = ocgantestdisjoint.main(opt)[1]
        print(res2)

        imax = np.argmax(np.array(auc3))
        opt.epochs = ran[imax]
        opt.istest=1
        res3  = ocgantestdisjoint.main(opt)[2]
        print(res3)

        imax = np.argmax(np.array(auc4))
        opt.epochs = ran[imax]
        opt.istest=1
        res4 = ocgantestdisjoint.main(opt)[3]
        print(res4)



        text_file.write("%s %s %s %s\n" % (str(res1), str(res2), str(res3), str(res4)))
	text_file.close()


'''
	print("AUC for criterion 1 (test): " + str(ocgantestdisjoint.main(opt)[0]))





#Pick best model w.r.t criterion 2
i = np.argmin(np.array(auc2))
opt.epochs = ran[i]
opt.istest=
print(ran[i])
print("AUC for criterion 2 (val): " + str(ocgantestdisjoint.main(opt)[1]))
opt.istest=1
print("AUC for criterion 2 (test): " + str(ocgantestdisjoint.main(opt)[1]))



#Pick best model w.r.t criterion 3
i = np.argmin(np.array(auc3))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 3 (val): " + str(ocgantestdisjoint.main(opt)[2]))
opt.istest=1
print("AUC for criterion 3 (test): " + str(ocgantestdisjoint.main(opt)[2]))



#Pick best model w.r.t criterion 4
i = np.argmin(np.array(auc4))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 4 (val): " + str(ocgantestdisjoint.main(opt)[3]))
opt.istest=1
print("AUC for criterion 4 (test): " + str(ocgantestdisjoint.main(opt)[3])'''
