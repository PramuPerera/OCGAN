# OneClassGAN

This repo contains code for OneClassGAN(OCGAN) developed by Pramuditha Perera with Ramesh Nallapati and Xiang Bing. To train a OneClass classifier use TrainNovelty.py. TestNovelty.py is used for testing. Basic functionality is as follows:

Prior to training include all data in class-specific sub folders as images. 
For protocol 1, all images of each class should be in a single folder. For example, in MNIST there should be folders called MNIST/0, MNIST/1, MNIST

For protocol 2, each class should have two sub-folders training and testing. Eg: MNIST/training/0, MNIST/training/1,…
And MNIST/testing/0, MNIST/testing/1,…

All data used in experiments can be found in the “noveltybuckety” s3 bucket.
Datasets used for both protocols has 3 subfolders: training, testing and all, where all contains all images of the given class (which can be used for protocol1).

Include a list of classes in a textfile dataset_folderlist.txt. Eg: for MNIST its MNIST_folderlist.txt

$ python2 TrainNovelty.py -—expname experiment1 -—dataset MNIST —-ngf 64 —-ndf 12 —-lambda  10  —-datapath ~/data  —-noisevar 0.2 —-classes 1 —-latent 16 

Where ~/data/MNIST/1 folder exists with intended class. 
 
To continue training from a saved epoch use --continueEpochFrom flag.

By default the network is set to be OCGAN. To use AE use —-ntype 1 argument. For ALOCC use —-ntype 2. 

Full list of arguments are as follows: 

  --expname EXPNAME     Name of the experiment
  --batch_size BATCH_SIZE
                        Batch size per iteration
  --epochs EPOCHS       Number of epochs for training
  --use_gpu USE_GPU     1 to use GPU
  --dataset DATASET     Specify the training dataset
  --lr LR               Base learning rate
  --ngf NGF             Number of base filters in Generator
  --ndf NDF             Number of base filters in Discriminator
  --beta1 BETA1         Parameter for Adam
  --lambda1 LAMBDA1     Weight of reconstruction loss
  --datapath DATAPATH   Data path
  --img_wd IMG_WD       Image width
  --img_ht IMG_HT       Image height
  --continueEpochFrom CONTINUEEPOCHFROM
                        Continue training from specified epoch
  --graphvis GRAPHVIS   1 to visualize the model
  --noisevar NOISEVAR   variance of noise added to input
  --depth DEPTH         Number of core layers in Generator/Discriminator
  --seed SEED           Seed generator. Use -1 for random.
  --append APPEND       Append discriminator input. 1 for true
  --istest ISTEST       Is this testing?. 1 for true
  --classes CLASSES     Name of training class. Keep blank for random
  --usegan USEGAN       set 1 for use gan loss.
  --useAE USEAE         set 1 for use AE.
  --latent LATENT       Dimension of the latent space.
  --ntype NTYPE         Novelty detector: 1 - AE 2 - ALOCC 3 - latentD 4 -
                        OCGAN
  —-protocol		Testing protocol used. 1: 80/20 protocol, 2: train/test protocol


For test the same model, use following command:
$ python2 TestNovelty.py -—expname experiment1 -—dataset MNIST —-ngf 64 —-ndf 12  —-datapath ~/data  —-noisevar 0.2  —-latent 16 

Area under the curve of ROC curve for following novelty detectors will be displayed:
1) D(De(En(x+n)))
2) D(x+n)
3) MSE (x, D(De(En(x+n))))
4) D(De(En(x)))


If the network type is AE only 3rd output will be calculated. Sample results will be saved in the outputs dir. In sample image results first correspond to input, ground truth and output.
