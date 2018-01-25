from torch.autograd import Variable
from utils import AverageMeter
from utils import accuracy
from utils import precision
from copy import deepcopy
import torch.nn as nn
import utils
import math
import time

from newlayers.BinActiveZ import Active

class Trainer():
    """
    Defines a trainer class that is used to train and evaluate models on various datasets
    """
    def __init__(self, model, criterion, optimizer, opt, logger):
        self.model = model
        self.realparams = deepcopy(model.parameters) # Used to save the real parameters during updates
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

    def train(self, trainloader, epoch, opt):
        """
        Trains the specified model for a single epoch on the training data
        """
        self.model.train()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.acc.reset()
        self.data_time.reset()
        self.batch_time.reset()

        end = time.time()
        for i, data in enumerate(trainloader, 0):

            if opt.binaryWeight:
                # Mean center and clamp weights of the conv layers within the range [binStart, binEnd]
                count = 1
                for m in self.model.modules():
                    if isinstance(m, nn.Conv2d):
                        if count >= opt.binStart and count <= opt.binEnd:
                            utils.meancenterConvParams(m)
                            utils.clampConvParams(m)
                        count += 1

                # Make a copy of the model parameters for use later
                self.realparams = deepcopy(self.model.parameters)
                # Binarize weights of the conv layers within the range [binStart, binEnd]
                count = 1
                for m in self.model.modules():
                    if isinstance(m, nn.Conv2d):
                        if count >= opt.binStart and count <= opt.binEnd:
                            utils.binarizeConvParams(m, opt.bnnModel)
                        count += 1

            self.optimizer.zero_grad()

            if opt.cuda:
                inputs, targets = data
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)

            inputs, targets = Variable(inputs), Variable(targets)

            self.data_time.update(time.time() - end)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            prec1, prec5 = precision(outputs.data, targets.data, topk=(1,5))
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)
            prec1, prec5 = prec1[0], prec5[0]

            loss.backward()

            if opt.binaryWeight:
                # Copy back the real weights stored earlier to all the conv layers
                current_parameters_list = list(self.realparams())
                current_count = 0
                for p in self.model.parameters():
                    p.data = current_parameters_list[current_count].data
                    current_count += 1

                # Apply gradient updates to all the layers
                count = 1
                for m in self.model.modules():
                    if isinstance(m, nn.Conv2d):
                        if count >= opt.binStart and count <= opt.binEnd:
                            utils.updateBinaryGradWeight(m, opt.bnnModel)
                        count += 1

            self.optimizer.step()

            inputs_size = inputs.size(0)
            self.losses.update(loss.data[0], inputs_size)
            self.acc.update(acc, inputs_size)
            self.top1.update(prec1, inputs_size)
            self.top5.update(prec5, inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f} ({batch_time.sum:.3f})\t'
                      'Data {data_time.avg:.3f} ({data_time.sum:.3f})\t'
                      'Loss {loss.avg:.3f}\t'
                      'Accuracy {acc.avg:.4f}\t'
                      'Prec@1 {top1.avg:.4f}\t'
                      'Prec@5 {top5.avg:.4f}'.format(
                       epoch, i, len(trainloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses,acc=self.acc,
                       top1=self.top1, top5=self.top5))

        # log to TensorBoard
        #if opt.tensorboard:
            #self.logger.scalar_summary('train_loss', self.losses.avg, epoch)
            #self.logger.scalar_summary('train_acc', self.top1.avg, epoch)

        print('Train: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Accuracy {acc.avg:.4f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               acc=self.acc, top1=self.top1, top5=self.top5))


class Validator():
    """
    Evaluates the specified model on the validation data
    """
    def __init__(self, model, criterion, opt, logger):

        self.model = model
        self.realparams = deepcopy(model.parameters)
        self.criterion = criterion
        self.logger = logger
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.acc = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()

        # Store the binarization losses for each of the layers for visualizations
        self.binarizationLosses = []
        self.binarizationLosses2 = []
        self.binarizationLosses3 = []
        self.itersforbin = 0

    def validate(self, valloader, epoch, opt):
        """
        Validates the specified model on the validation data
        """
        self.model.eval()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()
        self.acc.reset()
        self.data_time.reset()
        self.batch_time.reset()
        end = time.time()

        if opt.binaryWeight:
            # Mean center and clamp weights of the conv layers within the range [binStart, binEnd]
            count = 1
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    if count >= opt.binStart and count <= opt.binEnd:
                        utils.meancenterConvParams(m)
                        utils.clampConvParams(m)
                    count += 1

        if opt.binaryWeight:
            # Make a copy of the weights for use later
            self.realparams = deepcopy(self.model.parameters)

        if opt.binaryWeight:
            # Binarize weights of the conv layers within the range [binStart, binEnd]
            count = 1
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    if count >= opt.binStart and count <= opt.binEnd:
                        utils.binarizeConvParams(m, opt.bnnModel)
                    count += 1

        """
        A capture hook that allows us to analyze layer weights and produce a list of binarization losses based on the weights of each layer.
        Use on a trained FBin model which would allow us to produce a list of losses for each of the conv layers based on which a hybrid architecture can be designed.
        """
        def activ_forward_hook(self, inputs, outputs):
            feature_size = outputs.size()
            l2loss_tot = 0.0
            oneminusWloss_tot = 0.0
            w_tot = 0.0
            for batch_elem in range(feature_size[0]):
                l2loss_cur = (outputs[batch_elem]-inputs[0][batch_elem]).pow(2).sum().div(feature_size[1]*feature_size[2]*feature_size[3])
                l2loss_tot += l2loss_cur.data[0]
                inp_squared = (inputs[0][batch_elem].clamp(min=-1.0, max=1.0)).pow(2)
                oneminusWloss_cur = (1 - inp_squared).abs().sum().div(feature_size[1]*feature_size[2]*feature_size[3])
                oneminusWloss_tot += oneminusWloss_cur.data[0]
                w_tot += (inputs[0][batch_elem]).pow(2).sum().div(feature_size[1]*feature_size[2]*feature_size[3]).data[0]
            l2loss_tot /= feature_size[0]
            oneminusWloss_tot /= feature_size[0]
            w_tot /= feature_size[0]
            self.that.binarizationLosses[self.layer_num] += l2loss_tot
            self.that.binarizationLosses2[self.layer_num] += l2loss_tot/w_tot
            self.that.binarizationLosses3[self.layer_num] += oneminusWloss_tot

        layer_num = 0

        # Use hook to compute binarization losses if selected
        if opt.calculateBinarizationLosses:
            for m in self.model.modules():
                if isinstance(m, Active):
                    m.layer_num = layer_num
                    m.that = self
                    m.register_forward_hook(activ_forward_hook)
                    layer_num += 1
            for i in range(0, layer_num):
                self.binarizationLosses.append(0.0)
                self.binarizationLosses2.append(0.0)
                self.binarizationLosses3.append(0.0)

        for i, data in enumerate(valloader, 0):
            self.itersforbin += 1
            if opt.cuda:
                inputs, targets = data
                inputs = inputs.cuda(async=True)
                targets = targets.cuda(async=True)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
            # Arrange inputs for ten crop validation where each input is converted to 5 inputs
            if opt.tenCrop:
                inputs = inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2), inputs.size(3), inputs.size(4))

            self.data_time.update(time.time() - end)
            outputs = self.model(inputs)

            # Tweak outputs for ten crop validation where each output is duplicated based on the number of inputs
            if opt.tenCrop:
                outputs = outputs.view(outputs.size(0) // 10, 10, outputs.size(1)).sum(1).squeeze(1).div(10.0)

            loss = self.criterion(outputs, targets)
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)
            self.losses.update(loss.data[0], inputs[0].size(0))
            prec1, prec5 = precision(outputs.data, targets.data, topk=(1,5))
            prec1, prec5 = prec1[0], prec5[0]
            inputs_size = inputs.size(0)
            acc = accuracy(outputs.data.max(1)[1], targets.data, opt)

            self.acc.update(acc, inputs_size)
            self.top1.update(prec1, inputs_size)
            self.top5.update(prec5, inputs_size)

            # measure elapsed time
            self.batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.printfreq == 0 and opt.verbose == True:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       epoch, i, len(valloader), batch_time=self.batch_time,
                       data_time= self.data_time, loss=self.losses,
                       acc=self.acc, top1=self.top1, top5=self.top5))

        if opt.calculateBinarizationLosses:
            for i in range(0, layer_num):
                self.binarizationLosses[i] /= self.itersforbin
                self.binarizationLosses2[i] /= self.itersforbin
                self.binarizationLosses3[i] /= self.itersforbin
                self.binarizationLosses[i] = math.sqrt(self.binarizationLosses[i])
                self.binarizationLosses2[i] = math.sqrt(self.binarizationLosses2[i])
            print('Root Mean square error per convolution layer')
            print(self.binarizationLosses)
            print('Weight Normalized RMSE per convolution layer')
            print(self.binarizationLosses2)
            print('One minus W^2 error per convolution layer')
            print(self.binarizationLosses3)

        # Copy back the original set of weights
        if opt.binaryWeight:
            current_parameters_list = list(self.realparams())
            current_count = 0
            for p in self.model.parameters():
                p.data = current_parameters_list[current_count].data
                current_count += 1

        finalacc = self.acc.avg

        #if opt.tensorboard:
            #self.logger.scalar_summary('val_loss', self.losses.avg, epoch)
            #self.logger.scalar_summary('val_acc', self.top1.avg, epoch)

        print('Val: [{0}]\t'
              'Time {batch_time.sum:.3f}\t'
              'Data {data_time.sum:.3f}\t'
              'Loss {loss.avg:.3f}\t'
              'Accuracy {acc:.4f}\t'
              'Prec@1 {top1.avg:.4f}\t'
              'Prec@5 {top5.avg:.4f}\t'.format(
               epoch, batch_time=self.batch_time,
               data_time= self.data_time, loss=self.losses,
               acc=finalacc, top1=self.top1, top5=self.top5))

        return self.top1.avg
