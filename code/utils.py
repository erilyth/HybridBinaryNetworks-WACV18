import torch
import torch.nn as nn
from torch.nn import init
import copy
import random
import math
from PIL import Image
from torchvision import transforms

class AverageMeter():
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, opt):
    x = (output == target)
    x = sum(x)
    return x*1.0 / len(output)

def precision(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Invert(object):
    """
    Inverts images, used in the TU-Berlin loader
    """
    def __call__(self, img):
        img = 1.0 - img
        return img

class RandomRotate(object):
    """
    Rotates images randomly in a specified range
    """
    def __init__(self, rrange):
        self.rrange = rrange

    def __call__(self, img):
        size = img.size
        angle = random.randint(-self.rrange, self.rrange)
        img = img.rotate(angle, resample=Image.BICUBIC)
        img = img.resize(size, Image.ANTIALIAS)
        return img

class TenCrop(object):
    """
    Performs 10-crop validation on data
    """
    def __init__(self, size, opt):
        self.size = size
        self.opt = opt

    def __call__(self, img):
        centerCrop = transforms.CenterCrop(self.size)
        toPILImage = transforms.ToPILImage()
        toTensor = transforms.ToTensor()
        if self.opt.dataset == 'tuberlin':
            normalize = transforms.Normalize(mean=[0.06,], std=[0.93])
        if self.opt.dataset == 'sketchyrecognition':
            normalize = transforms.Normalize(mean=[0.0465,], std=[0.9])
        w, h = img.size(2), img.size(1)
        temp_output = []
        output = torch.FloatTensor(10, img.size(0), self.size, self.size)
        img = toPILImage(img)
        for img_cur in [img, img.transpose(Image.FLIP_LEFT_RIGHT)]:
            temp_output.append(centerCrop(img_cur))
            temp_output.append(img_cur.crop([0, 0, self.size, self.size]))
            temp_output.append(img_cur.crop([w-self.size, 0, w, self.size]))
            temp_output.append(img_cur.crop([0, h-self.size, self.size, h]))
            temp_output.append(img_cur.crop([w-self.size, h-self.size, w, h]))

        for img_idx in range(10):
            img_cur = temp_output[img_idx]
            img_cur = toTensor(img_cur)
            img_cur = normalize(img_cur)
            output[img_idx] = img_cur.view(img_cur.size(0), img_cur.size(1), img_cur.size(2))

        return output

def binarizeConvParams(convNode, bnnModel):
    """
    Binarizes a given Conv layer
    """
    s = convNode.weight.data.size()
    n = s[1]*s[2]*s[3]
    if bnnModel:
      convNode.weight.data[convNode.weight.data.eq(0)] = -1e-6
      convNode.weight.data = convNode.weight.data.sign()
    else:
      m = convNode.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n)
      orig_alpha = copy.deepcopy(m[0][0][0])

      convNode.weight.data[convNode.weight.data.eq(0)] = -1e-6
      # Used for visualizing a specific set of weight filters
      cur = convNode.weight.data[6].sign()
      cur[cur.lt(0)] = 0

      convNode.weight.data = convNode.weight.data.sign().mul(m.repeat(1, s[1], s[2], s[3]))
      # Return values for visualizations
      return cur.sum(), orig_alpha

def updateBinaryGradWeight(convNode, bnnModel):
    """
    Assign gradients to binarized Conv layer used for weight updates
    """
    s = convNode.weight.data.size()
    n = s[1]*s[2]*s[3]
    m = convNode.weight.data.clone()
    if bnnModel:
        m = convNode.weight.data.clone().fill(1)
        m[convNode.weight.data.le(-1)] = 0
        m[convNode.weight.data.ge(1)] = 0
        m = torch.mul(1 - 1.0/s[1])
    else:
        m = convNode.weight.data.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).repeat(1, s[1], s[2], s[3])
        m[convNode.weight.data.le(-1)] = 0
        m[convNode.weight.data.ge(1)] = 0
        m = torch.add(m, 1.0/n)
        m = torch.mul(m, 1.0 - 1.0/s[1])
        m = torch.mul(m, n)
    convNode.weight.grad.data.mul_(m)

def meancenterConvParams(convNode):
    """
    Mean-center the weights of the given Conv Node
    """
    s = convNode.weight.data.size()
    negMean = torch.mul(convNode.weight.data.mean(1,keepdim=True),-1).repeat(1,1,1,1)
    #print(negMean.size())
    negMean = negMean.repeat(1, s[1], 1, 1)
    #print(negMean.size(),convNode.weight.data.size())
    convNode.weight.data.add_(negMean)

def clampConvParams(convNode):
    """
    Clamp the weights of a given Conv layer
    """
    convNode.weight.data.clamp_(-1, 1)

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    epoch = copy.deepcopy(epoch)
    lr = opt.maxlr
    wd = opt.weightDecay
    if opt.learningratescheduler == 'imagenetscheduler':
        if epoch >= 1 and epoch <= 18:
            lr = 1e-3
            wd = 5e-5
        elif epoch >= 19 and epoch <= 29:
            lr = 5e-4
            wd = 5e-5
        elif epoch >= 30 and epoch <= 43:
            lr = 1e-4
            wd = 0
        elif epoch >= 44 and epoch <= 52:
            lr = 5e-5
            wd = 0
        elif epoch >= 53:
            lr = 2e-5
            wd = 0
        if opt.optimType=='sgd':
            lr *= 10
        opt.lr = lr
        opt.weightDecay = wd
    if opt.learningratescheduler == 'decayscheduler':
        while epoch >= opt.decayinterval:
            lr = lr/opt.decaylevel
            epoch = epoch - opt.decayinterval
        lr = max(lr,opt.minlr)
        opt.lr = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = wd

def get_mean_and_std(dataloader):
    """
    Compute the mean and std value of dataset
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    len_dataset = 0
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        len_dataset += 1
        for i in range(len(inputs[0])):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len_dataset)
    std.div_(len_dataset)
    return mean, std

def weights_init(model, opt):
    """
    Perform weight initializations
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if m.affine == True:
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            c =  math.sqrt(2.0 / m.weight.data.size(1));
            if m.bias is not None:
                init.constant(m.bias, 0)
