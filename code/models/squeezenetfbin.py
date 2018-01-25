import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from newlayers.BinActiveZ import Active


__all__ = ['SqueezeNet', 'Net']


model_urls = {
    'squeezenet': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.activ1 = Active()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, bias=False)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.bn2 = nn.BatchNorm2d(squeeze_planes)
        self.activ2 = Active()
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1, bias=False)
        self.expand1x1_activation = nn.ReLU(inplace=True)

        self.bn3 = nn.BatchNorm2d(squeeze_planes)
        self.activ3 = Active()
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1, bias=False)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(self.activ1(self.bn1(x))))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(self.activ2(self.bn2(x)))),
            self.expand3x3_activation(self.expand3x3(self.activ3(self.bn3(x))))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, nClasses):
        super(SqueezeNet, self).__init__()
        self.num_classes = nClasses
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256, 256),
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        x = x.view(-1, self.num_classes)

        return F.log_softmax(x)


def Net(nClasses, pretrained=False, ):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(nClasses)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet']))
    return model
