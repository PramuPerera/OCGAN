import options
import twoDtest
import numpy as np

opt = options.test_options()
opt.istest = 0
#First use the validation set to pick best model
text_file = open(opt.dataset + "_progress.txt", "w")
auc1 = []
auc2=[]
auc3=[] 
auc4=[]
ran = range(0,400,10)
for i in ran:
    opt.epochs = i
    roc_auc = twoDtest.main(opt)
    print(roc_auc)
    auc1+=roc_auc[0]
    auc2+=roc_auc[1]
    auc3+=roc_auc[2]
    auc4+=roc_auc[3]
    text_file.write("%s %s %s %s %s\n" % (str(i), str(roc_auc[0]), str(roc_auc[1]),str(roc_auc[2]),str(roc_auc[3])))
text_file.close()
print('Validation Done')

#Pick best model w.r.t criterion 1
i = np.argmax(np.array(auc1))
opt.epochs = ran[i]
opt.istest=0
print("AUC for criterion 1 (val): " + twoDtest.main(opt))
opt.istest=1
print("AUC for criterion 1 (test): " + twoDtest.main(opt))



#Pick best model w.r.t criterion 2
i = np.argmax(np.array(auc2))
opt.epochs = ran[i]
opt.istest=0
print("AUC for criterion 2 (val): " + twoDtest.main(opt))
opt.istest=1
print("AUC for criterion 2 (test): " + twoDtest.main(opt))



#Pick best model w.r.t criterion 3
i = np.argmax(np.array(auc3))
opt.epochs = ran[i]
opt.istest=0
print("AUC for criterion 3 (val): " + twoDtest.main(opt))
opt.istest=1
print("AUC for criterion 3 (test): " + twoDtest.main(opt))



#Pick best model w.r.t criterion 4
i = np.argmax(np.array(auc4))
opt.epochs = ran[i]
opt.istest=0
print("AUC for criterion 4 (val): " + twoDtest.main(opt))
opt.istest=1
print("AUC for criterion 4 (test): " + twoDtest.main(opt))
