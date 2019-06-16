from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
import random
from random import shuffle
import dataloaderiter as dload
import load_image
import visual
import models
from datetime import datetime
import time
import logging
import argparse
import options


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


def trainadnov(opt, train_data, val_data, ctx, networks):

    netEn = networks[0]
    netDe = networks[1]
    netD = networks[2]
    netD2 = networks[3]
    netDS = networks[4]
    trainerEn = networks[5]
    trainerDe = networks [6]
    trainerD =networks[7]
    trainerD2 = networks[8]
    trainerSD = networks[9]
    cep = opt.continueEpochFrom
    epochs = opt.epochs
    lambda1 = opt.lambda1
    batch_size = opt.batch_size
    expname = opt.expname
    append = opt.append
    text_file = open(expname + "_trainloss.txt", "w")
    text_file.close()
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    metric = mx.metric.CustomMetric(facc)
    metricl = mx.metric.CustomMetric(facc)
    metricStrong = mx.metric.CustomMetric(facc)
    metric2 = mx.metric.MSE()
    loss_rec_G2 =[]
    acc2_rec = []
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    lr = 2.0 * batch_size
    logging.basicConfig(level=logging.DEBUG)
    if cep == -1:
        cep = 0
    else:
        netEn.load_params('checkpoints/' + opt.expname + '_' + str(cep) + '_En.params', ctx=ctx)
        netDe.load_params('checkpoints/' + opt.expname + '_' + str(cep) + '_De.params', ctx=ctx)
        netD.load_params('checkpoints/' + opt.expname + '_' + str(cep) + '_D.params', ctx=ctx)
        netD2.load_params('checkpoints/' + opt.expname + '_' + str(cep) + '_D2.params', ctx=ctx)
        netDS.load_params('checkpoints/' + opt.expname + '_' + str(cep) + '_SD.params', ctx=ctx)
    for epoch in range(cep + 1, epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)
            fake_latent = netEn(real_in)
            mu = nd.random.uniform( low=-1, high=1, shape=fake_latent.shape, ctx=ctx)
            real_latent = nd.random.uniform(low=-1, high=1, shape=fake_latent.shape, ctx=ctx)
            fake_out = netDe(fake_latent)
            fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
            if epoch > 150:  # negative mining
                mu = nd.random.uniform(low=-1, high=1, shape=fake_latent.shape, ctx=ctx)
                mu.attach_grad()
                for ep2 in range(1): # doing single gradient step
                    with autograd.record():
                        eps2 = nd.tanh(mu)
                        rec_output = netDS(netDe(eps2))
                        fake_label = nd.zeros(rec_output.shape, ctx=ctx)
                        errGS = GAN_loss(rec_output, fake_label)
                        errGS.backward()
                    mu -= lr / mu.shape[0] * mu.grad # Update mu with SGD
            eps2 = nd.tanh(mu)
            with autograd.record():
                # Train with fake image
                output = netD(fake_concat)
                output2 = netD2(fake_latent)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                fake_latent_label = nd.zeros(output2.shape, ctx=ctx)
                eps = nd.random.uniform(low=-1, high=1, shape=fake_latent.shape, ctx=ctx)
                rec_output = netD(netDe(eps))
                errD_fake = GAN_loss(rec_output, fake_label)
                errD_fake2 = GAN_loss(output, fake_label)
                errD2_fake = GAN_loss(output2, fake_latent_label)
                metric.update([fake_label, ], [rec_output, ])
                metric2.update([fake_latent_label, ], [output2, ])
                real_concat = nd.concat(real_in, real_out, dim=1) if append else real_out
                output = netD(real_concat)
                output2 = netD2(real_latent)
                real_label = nd.ones(output.shape, ctx=ctx)
                real_latent_label = nd.ones(output2.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD2_real = GAN_loss(output2, real_latent_label)
                errD = (errD_real + errD_fake) * 0.5
                errD2 = (errD2_real + errD2_fake) * 0.5
                totalerrD = errD + errD2
                totalerrD.backward()
            metric.update([real_label, ], [output, ])
            metric2.update([real_latent_label, ], [output2, ])
            trainerD.step(batch.data[0].shape[0])
            trainerD2.step(batch.data[0].shape[0])
            with autograd.record():
                # Train classifier
                strong_output = netDS(netDe(eps))
                strong_real = netDS(fake_concat)
                errs1 = GAN_loss(strong_output, fake_label)
                errs2 = GAN_loss(strong_real, real_label)
                metricStrong.update([fake_label, ], [strong_output, ])
                metricStrong.update([real_label, ], [strong_real, ])
                strongerr = 0.5 * (errs1 + errs2)
                strongerr.backward()
            trainerSD.step(batch.data[0].shape[0])
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                rec_output = netD(netDe(eps2))
                fake_latent = (netEn(real_in))
                output2 = netD2(fake_latent)
                fake_out = netDe(fake_latent)
                fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                real_latent_label = nd.ones(output2.shape, ctx=ctx)
                errG2 = GAN_loss(rec_output, real_label)
                errR = L1_loss(real_out, fake_out) * lambda1
                errG = 10.0 * GAN_loss(output2, real_latent_label) + errG2 + errR
                errG.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
            loss_rec_G2.append(nd.mean(errG2).asscalar())
            loss_rec_G.append(nd.mean(nd.mean(errG)).asscalar() - nd.mean(errG2).asscalar() - nd.mean(errR).asscalar())
            loss_rec_D.append(nd.mean(errD).asscalar())
            loss_rec_R.append(nd.mean(errR).asscalar())
            loss_rec_D2.append(nd.mean(errD2).asscalar())
            _, acc2 = metric2.get()
            name, acc = metric.get()
            acc_rec.append(acc)
            acc2_rec.append(acc2)

            # Print log infomation every ten batches
            if iter % 10 == 0:
                _, acc2 = metric2.get()
                name, acc = metric.get()
            _, accStrong = metricStrong.get()
            logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
            logging.info(
                'discriminator loss = %f, D2 loss = %f, generator loss = %f, G2 loss = %f, SD loss = %f,  D acc = %f , D2 acc = %f, DS acc = %f, reconstruction error= %f  at iter %d epoch %d'
                % (nd.mean(errD).asscalar(), nd.mean(errD2).asscalar(),
                   nd.mean(errG - errG2 - errR).asscalar(), nd.mean(errG2).asscalar(), nd.mean(strongerr).asscalar(), acc, acc2,
                   accStrong, nd.mean(errR).asscalar(), iter, epoch))
            iter = iter + 1
            btic = time.time()
            name, acc = metric.get()
            _, acc2 = metric2.get()
            metric.reset()
            metric2.reset()
            train_data.reset()
            metricStrong.reset()

            logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
            logging.info('time: %f' % (time.time() - tic))
            if epoch % 5 == 0:
                filename = "checkpoints/" + expname + "_" + str(epoch) + "_D.params"
                netD.save_params(filename)
                filename = "checkpoints/" + expname + "_" + str(epoch) + "_D2.params"
                netD2.save_params(filename)
                filename = "checkpoints/" + expname + "_" + str(epoch) + "_En.params"
                netEn.save_params(filename)
                filename = "checkpoints/" + expname + "_" + str(epoch) + "_De.params"
                netDe.save_params(filename)
                filename = "checkpoints/" + expname + "_" + str(epoch) + "_SD.params"
                netDS.save_params(filename)
                val_data.reset()
                text_file = open(expname + "_validtest.txt", "a")
                for vbatch in val_data:
                    real_in = vbatch.data[0].as_in_context(ctx)
                    real_out = vbatch.data[1].as_in_context(ctx)
                    fake_latent = netEn(real_in)
                    y = netDe(fake_latent)
                    fake_out = y
                    metricMSE.update([fake_out, ], [real_out, ])
                _, acc2 = metricMSE.get()
                text_file.write("%s %s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2), str(accStrong)))
                metricMSE.reset()
    return [loss_rec_D, loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2, loss_rec_G2, acc2_rec]


def trainAE(opt, train_data, val_data, ctx, networks):

    netEn = networks[0]
    netDe = networks[1]
    trainerEn = networks[5]
    trainerDe = networks[6]
    epochs = opt.epochs
    batch_size = opt.batch_size
    expname = opt.expname
    text_file = open(expname + "_trainloss.txt", "w")
    text_file.close()
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    L1_loss = gluon.loss.L2Loss()
    metric2 = mx.metric.MSE()
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)
            with autograd.record():
                fake_out = netDe(netEn(real_in))
                errR = L1_loss(real_out, fake_out)
                errR.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
        loss_rec_R.append(nd.mean(errR).asscalar())

        if iter % 10 == 0:
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('reconstruction error= %f at iter %d epoch %d'
                            % (nd.mean(errR).asscalar(), iter, epoch))
        iter = iter + 1
        btic = time.time()
        text_tl = open(expname + "_trainloss.txt", "a")
        text_tl.write('%f %f %f %f %f %f %f ' % (0, 0, 0, 0, 0, nd.mean(errR).asscalar(), epoch))
        text_file.close()
        train_data.reset()
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            fake_img1 = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                real_in = vbatch.data[0].as_in_context(ctx)
                real_out = vbatch.data[1].as_in_context(ctx)
                fake_out = netDe(netEn(real_in))
                metric2.update([fake_out, ], [real_out, ])
                _, acc2 = metric2.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metric2.reset()
            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img = nd.concat(fake_img1, fake_img2, fake_img3, fake_img1T, fake_img2T, fake_img3T, dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
            text_file.close()
    return([loss_rec_D,loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2])


def traincvpr18(opt, train_data, val_data, ctx, networks):

    netEn = networks[0]
    netDe = networks[1]
    netD = networks[2]
    trainerEn = networks[5]
    trainerDe = networks [6]
    trainerD =networks[7]
    epochs = opt.epochs
    lambda1 = opt.lambda1
    batch_size = opt.batch_size
    expname = opt.expname
    append = opt.append
    text_file = open(expname + "_trainloss.txt", "w")
    text_file.close()
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    metric = mx.metric.CustomMetric(facc)
    metricl = mx.metric.CustomMetric(facc)
    metric2 = mx.metric.MSE()
    loss_rec_G2 =[]
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)
            fake_latent = netEn(real_in)
            fake_out = netDe(fake_latent)
            fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history imagesi
                output = netD(fake_concat)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label, ], [output, ])
                real_concat = nd.concat(real_in, real_out, dim=1) if append else real_out
                output = netD(real_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                metric.update([real_label, ], [output, ])
            trainerD.step(batch.data[0].shape[0])
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_latent = (netEn(real_in))
                fake_out = netDe(fake_latent)
                fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                errR = L1_loss(real_out, fake_out)
                errG.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
        loss_rec_G.append(nd.mean(errG).asscalar()-nd.mean(errR).asscalar()*lambda1)
        loss_rec_D.append(nd.mean(errD).asscalar())
        loss_rec_R.append(nd.mean(errR).asscalar())
        name, acc = metric.get()
        acc_rec.append(acc)
        # Print log infomation every ten batches
        if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f , reconstruction error= %f at iter %d epoch %d'
                            % (nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, nd.mean(errR).asscalar(), iter, epoch))
        iter = iter + 1
        btic = time.time()

        name, acc = metric.get()
        _, acc2 = metricl.get()
        text_tl = open(expname + "_trainloss.txt", "a")
        text_tl.write('%f %f %f %f %f %f %f ' % (nd.mean(errD).asscalar(), 0,
                    nd.mean(errG).asscalar(), acc, 0, nd.mean(errR).asscalar(), epoch))
        text_file.close()
        metricl.reset()
        metric.reset()
        train_data.reset()

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            fake_img1 = nd.concat(real_in[0], real_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1], real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2], real_out[2], fake_out[2], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                real_in = vbatch.data[0].as_in_context(ctx)
                real_out = vbatch.data[1].as_in_context(ctx)
                fake_latent= netEn(real_in)
                y = netDe(fake_latent)
                fake_out = y
                metric2.update([fake_out, ], [real_out, ])
                _, acc2 = metric2.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metric2.reset()
            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img = nd.concat(fake_img1,fake_img2, fake_img3, fake_img1T, fake_img2T, fake_img3T, dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
            text_file.close()
    return [loss_rec_D,loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2]

