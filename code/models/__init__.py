import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import models.alexnet as alexnet
import models.alexnetfbin as alexnetfbin
import models.alexnethybrid as alexnethybrid
import models.alexnethybridv2 as alexnethybridv2
import models.alexnetwbin as alexnetwbin
import models.googlenet as googlenet
import models.googlenetfbin as googlenetfbin
import models.googlenetwbin as googlenetwbin
import models.mobilenet as mobilenet
import models.resnet as resnet
import models.resnetfbin as resnetfbin
import models.resnethybrid as resnethybrid
import models.resnethybridv2 as resnethybridv2
import models.resnethybridv3 as resnethybridv3
import models.resnetwbin as resnetwbin
import models.sketchanet as sketchanet
import models.sketchanetfbin as sketchanetfbin
import models.sketchanethybrid as sketchanethybrid
import models.sketchanethybridv2 as sketchanethybridv2
import models.sketchanetwbin as sketchanetwbin
import models.squeezenet as squeezenet
import models.squeezenetfbin as squeezenetfbin
import models.squeezenethybrid as squeezenethybrid
import models.squeezenethybridv2 as squeezenethybridv2
import models.squeezenethybridv3 as squeezenethybridv3
import models.squeezenetwbin as squeezenetwbin
import models.bincifar as bincifar
import models.densenet as densenet
import models.vgg as vgg
import models.googlenetexpanded as googlenet_expanded

import utils
import os

"""
Initialize and return the model, criterion and the optimizer
"""
def setup(model, opt):

    if opt.criterion == "nllLoss":
        criterion = nn.NLLLoss().cuda()

    if opt.optimType == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum, nesterov = opt.nesterov, weight_decay = opt.weightDecay)
    elif opt.optimType == 'adam':
        optimizer = optim.Adam(model.parameters(), lr = opt.maxlr, weight_decay = opt.weightDecay)

    if opt.weight_init:
        utils.weights_init(model, opt)

    return model, criterion, optimizer

"""
Save the current state as a checkpoint
"""
def save_checkpoint(opt, model, optimizer, best_acc, epoch):

    state = {
        'epoch': epoch + 1,
        'arch': opt.model_def,
        'state_dict': model.state_dict(),
        'best_prec1': best_acc,
        'optimizer' : optimizer.state_dict(),
    }
    filename = "savedmodels/" + opt.name + ".pth.tar"

    torch.save(state, filename)

"""
Resume from a given checkpoint
"""
def resumer(opt, model, optimizer):

    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))

        return model, optimizer, opt, best_prec1

"""
Load a model from the specified opts
"""
def load_model(opt):
    if opt.pretrained_file != "":
        model = torch.load(opt.pretrained_file)
    else:
        if opt.model_def == 'alexnet':
            model = alexnet.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'bincifar':
            model = bincifar.Net()
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'densenet':
            model = densenet.DenseNet3(16, 10)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'alexnetfbin':
            model = alexnetfbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'alexnethybrid':
            model = alexnethybrid.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'alexnethybridv2':
            model = alexnethybridv2.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'alexnetwbin':
            model = alexnetwbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'googlenet':
            model = googlenet.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'googlenet_expanded':
            model = googlenet_exp.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'googlenetfbin':
            model = googlenetfbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'googlenetwbin':
            model = googlenetwbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'mobilenet':
            model = mobilenet.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnet18':
            model = resnet.resnet18(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnetfbin18':
            model = resnetfbin.resnet18(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnethybrid18':
            model = resnethybrid.resnet18(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnethybridv218':
            model = resnethybridv2.resnet18(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnethybridv318':
            model = resnethybridv3.resnet18(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'resnetwbin18':
            model = resnetwbin.resnet18(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'sketchanet':
            model = sketchanet.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'sketchanetfbin':
            model = sketchanetfbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'sketchanethybrid':
            model = sketchanethybrid.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'sketchanethybridv2':
            model = sketchanethybridv2.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'sketchanetwbin':
            model = sketchanetwbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'squeezenet':
            model = squeezenet.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'squeezenetfbin':
            model = squeezenetfbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'squeezenethybrid':
            model = squeezenethybrid.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'squeezenethybridv2':
            model = squeezenethybridv2.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'squeezenethybridv3':
            model = squeezenethybridv3.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'squeezenetwbin':
            model = squeezenetwbin.Net(opt.nClasses)
            if opt.cuda:
                model = model.cuda()

        elif opt.model_def == 'vgg16_bncifar':
            model = vgg.vgg16_bn()
            if opt.cuda:
                model = model.cuda()

    return model
