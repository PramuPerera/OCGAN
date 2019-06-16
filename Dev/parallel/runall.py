import options
import ocgantestdisjoint

opt = options.test_options()
text_file = open(opt.dataset + "_progress.txt", "w")
for i in range(0,1000,10):
    opt.epochs = i
    roc_auc = ocgantestdisjoint.main(opt)
    print(roc_auc)
    auc1=roc_auc[0]
    auc2=roc_auc[1]
    auc3=roc_auc[2]
    auc4=roc_auc[3]
    text_file.write("%s %s %s %s %s\n" % (str(i), str(auc1),str(auc2),str(auc3),str(auc4)))
text_file.close()

