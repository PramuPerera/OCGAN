import argparse
def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", default="expce", help="Name of the experiment")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size per iteration")
    parser.add_argument("--epochs", default=201, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--use_gpu", default=1, type=int,  help="1 to use GPU  ")
    parser.add_argument("--dataset", default="Caltech256",
                        help="Specify the training dataset  ")
    parser.add_argument("--lr", default="0.0002", type=float, help="Base learning rate")
    parser.add_argument("--ngf", default=64, type=int, help="Number of base filters in Generator")
    parser.add_argument("--ndf", default=8, type=int, help="Number of base filters in Discriminator")
    parser.add_argument("--beta1", default=0.5, type=float, help="Parameter for Adam")
    parser.add_argument("--lambda1", default=500, type=float, help="Weight of reconstruction loss")
    parser.add_argument("--datapath", default='/users/pramudi/Documents/data/', help="Data path")
    parser.add_argument("--img_wd", default=61, type=int, help="Image width")
    parser.add_argument("--img_ht", default=61, type=int, help="Image height")
    parser.add_argument("--continueEpochFrom", default=-1,
                        help="Continue training from specified epoch")
    parser.add_argument("--noisevar", default=0.02, type=float, help="variance of noise added to input")
    parser.add_argument("--depth", default=3, type=int, help="Number of core layers in Generator/Discriminator")
    parser.add_argument("--seed", default=-1, type=float, help="Seed generator. Use -1 for random.")
    parser.add_argument("--append", default=0, type=int, help="Append discriminator input. 1 for true")
    parser.add_argument("--classes", default="", help="Name of training class. Keep blank for random")
    parser.add_argument("--latent", default=16, type=int,  help="Dimension of the latent space.")
    parser.add_argument("--ntype", default=4, type=int, help="Novelty detector: 1 - AE 2 - ALOCC 3 - latentD 4 - OCGAN")
    parser.add_argument("--protocol", default=1, type=int, help="1 : 80/20 split, 2 : Train / Test split")
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
    if args.append == 1:
        args.append = True
    else:
        args.append = False
    return args


def test_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", default="expce", help="Name of the experiment")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size per iteration")
    parser.add_argument("--epochs", default=201, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--use_gpu", default=1, type=int,  help="1 to use GPU  ")
    parser.add_argument("--dataset", default="Caltech256",
                        help="Specify the training dataset  ")
    parser.add_argument("--ngf", default=64, type=int, help="Number of base filters in Generator")
    parser.add_argument("--ndf", default=8, type=int, help="Number of base filters in Discriminator")
    parser.add_argument("--datapath", default='/users/pramudi/Documents/data/', help="Data path")
    parser.add_argument("--img_wd", default=61, type=int, help="Image width")
    parser.add_argument("--img_ht", default=61, type=int, help="Image height")
    parser.add_argument("--latent", default=4096, type=int,  help="Dimension of the latent space.")
    parser.add_argument("--depth", default=3, type=int, help="Number of core layers in Generator/Discriminator")
    parser.add_argument("--noisevar", default=0.02, type=float, help="variance of noise added to input")
    parser.add_argument("--istest", default=1, type=float, help="if test set 1, otherwise validation")
    parser.add_argument("--append", default=0, type=int, help="Append discriminator input. 1 for true")
    parser.add_argument("--isvalidation", default=0, type=float, help="Pass through training set. Utility for development.")
    parser.add_argument("--usegan", default=1, type=int,  help="set 1 for use gan loss.")
    parser.add_argument("--ntype", default=4, type=int, help="Novelty detector: 1 - AE 2 - ALOCC 3 - latentD 4 - adnov")
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
    if args.usegan == 1:
        args.usegan = True
    else:
        args.usegan = False    
    if args.istest == 1:
        args.istest = True
    else:
        args.istest = False
    if args.append == 1:
        args.append = True
    else:
        args.append = False
    if args.isvalidation == 1:
        args.isvalidation = True
    else:
        args.isvalidation = False
    return args
