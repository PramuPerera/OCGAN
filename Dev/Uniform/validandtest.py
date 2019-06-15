import options
import ocgantestdisjoint
import numpy as np
import random
random.seed(1000)
opt = options.test_options()
opt.istest = 0
#First use the validation set to pick best model
text_file = open(opt.dataset + "_progress.txt", "w")
auc1 = []
auc2=[]
auc3=[]
auc4=[]
ran = range(0,500,10)
for i in ran:
    opt.epochs = i
    roc_auc = ocgantestdisjoint.main(opt)
    print(roc_auc)
    auc1.append(roc_auc[0])
    auc2.append(roc_auc[1])
    auc3.append(roc_auc[2])
    auc4.append(roc_auc[3])
    text_file.write("%s %s %s %s %s\n" % (str(i), str(roc_auc[0]), str(roc_auc[1]),str(roc_auc[2]),str(roc_auc[3])))
text_file.close()
print('Validation Done')

#Pick best model w.r.t criterion 1
i = np.argmax(np.array(auc1))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 1 (val): " + str(ocgantestdisjoint.main(opt)[0]))
opt.istest=1
print("AUC for criterion 1 (test): " + str(ocgantestdisjoint.main(opt)[0]))



#Pick best model w.r.t criterion 2
i = np.argmax(np.array(auc2))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 2 (val): " + str(ocgantestdisjoint.main(opt)[1]))
opt.istest=1
print("AUC for criterion 2 (test): " + str(ocgantestdisjoint.main(opt)[1]))



#Pick best model w.r.t criterion 3
i = np.argmax(np.array(auc3))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 3 (val): " + str(ocgantestdisjoint.main(opt)[2]))
opt.istest=1
print("AUC for criterion 3 (test): " + str(ocgantestdisjoint.main(opt)[2]))



#Pick best model w.r.t criterion 4
i = np.argmax(np.array(auc4))
opt.epochs = ran[i]
opt.istest=0
print(ran[i])
print("AUC for criterion 4 (val): " + str(ocgantestdisjoint.main(opt)[3]))
opt.istest=1
print("AUC for criterion 4 (test): " + str(ocgantestdisjoint.main(opt)[3]))
