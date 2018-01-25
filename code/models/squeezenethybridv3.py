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


class SqueezeNet(nn.Module):

    def __init__(self, nClasses):
        super(SqueezeNet, self).__init__()
        self.num_classes = nClasses
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv21 = nn.Conv2d(96, 16, kernel_size=1)
        self.bn21 = nn.BatchNorm2d(16)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(16, 64, kernel_size=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.relu22 = nn.ReLU(inplace=True)
        self.conv23 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn23 = nn.BatchNorm2d(64)
        self.relu23 = nn.ReLU(inplace=True)

        self.conv31 = nn.Conv2d(128, 16, kernel_size=1, bias=False)
        self.bn31 = nn.BatchNorm2d(16)
        self.relu31 = nn.ReLU(inplace=True)
        self.conv32 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.bn32 = nn.BatchNorm2d(64)
        self.relu32 = nn.ReLU(inplace=True)
        self.bn33 = nn.BatchNorm2d(16)
        self.activ33 = Active()
        self.conv33 = nn.Conv2d(16, 64, kernel_size=3, padding=1, bias=False)
        self.relu33 = nn.ReLU(inplace=True)

        self.conv41 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.bn41 = nn.BatchNorm2d(32)
        self.relu41 = nn.ReLU(inplace=True)
        self.conv42 = nn.Conv2d(32, 128, kernel_size=1, bias=False)
        self.bn42 = nn.BatchNorm2d(128)
        self.relu42 = nn.ReLU(inplace=True)
        self.bn43 = nn.BatchNorm2d(32)
        self.activ43 = Active()
        self.conv43 = nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False)
        self.relu43 = nn.ReLU(inplace=True)

        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv61 = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        self.bn61 = nn.BatchNorm2d(32)
        self.relu61 = nn.ReLU(inplace=True)
        self.conv62 = nn.Conv2d(32, 128, kernel_size=1, bias=False)
        self.bn62 = nn.BatchNorm2d(128)
        self.relu62 = nn.ReLU(inplace=True)
        self.bn63 = nn.BatchNorm2d(32)
        self.activ63 = Active()
        self.conv63 = nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False)
        self.relu63 = nn.ReLU(inplace=True)

        self.conv71 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.bn71 = nn.BatchNorm2d(48)
        self.relu71 = nn.ReLU(inplace=True)
        self.conv72 = nn.Conv2d(48, 192, kernel_size=1, bias=False)
        self.bn72 = nn.BatchNorm2d(192)
        self.relu72 = nn.ReLU(inplace=True)
        self.bn73 = nn.BatchNorm2d(48)
        self.activ73 = Active()
        self.conv73 = nn.Conv2d(48, 192, kernel_size=3, padding=1, bias=False)
        self.relu73 = nn.ReLU(inplace=True)

        self.conv81 = nn.Conv2d(384, 48, kernel_size=1, bias=False)
        self.bn81 = nn.BatchNorm2d(48)
        self.relu81 = nn.ReLU(inplace=True)
        self.conv82 = nn.Conv2d(48, 192, kernel_size=1, bias=False)
        self.bn82 = nn.BatchNorm2d(192)
        self.relu82 = nn.ReLU(inplace=True)
        self.bn83 = nn.BatchNorm2d(48)
        self.activ83 = Active()
        self.conv83 = nn.Conv2d(48, 192, kernel_size=3, padding=1, bias=False)
        self.relu83 = nn.ReLU(inplace=True)

        self.conv91 = nn.Conv2d(384, 64, kernel_size=1, bias=False)
        self.bn91 = nn.BatchNorm2d(64)
        self.relu91 = nn.ReLU(inplace=True)
        self.conv92 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.bn92 = nn.BatchNorm2d(256)
        self.relu92 = nn.ReLU(inplace=True)
        self.bn93 = nn.BatchNorm2d(64)
        self.activ93 = Active()
        self.conv93 = nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False)
        self.relu93 = nn.ReLU(inplace=True)

        self.maxpool10 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv111 = nn.Conv2d(512, 64, kernel_size=1, bias=False)
        self.bn111 = nn.BatchNorm2d(64)
        self.relu111 = nn.ReLU(inplace=True)
        self.conv112 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.bn112 = nn.BatchNorm2d(256)
        self.relu112 = nn.ReLU(inplace=True)
        self.conv113 = nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False)
        self.bn113 = nn.BatchNorm2d(256)
        self.relu113 = nn.ReLU(inplace=True)

        self.drop121 = nn.Dropout(p=0.2)
        self.conv121 = nn.Conv2d(512, self.num_classes, kernel_size=1, bias=False)
        self.relu121 = nn.ReLU(inplace=True)
        self.avgpool121 = nn.AvgPool2d(13)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.conv121:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu21(x)
        x1 = self.conv22(x)
        x1 = self.bn22(x1)
        x1 = self.relu22(x1)
        x2 = self.conv23(x)
        x2 = self.bn23(x2)
        x2 = self.relu23(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv31(x)
        x = self.bn31(x)
        x = self.relu31(x)
        x1 = self.conv32(x)
        x1 = self.bn32(x1)
        x1 = self.relu32(x1)
        x2 = self.bn33(x)
        x2 = self.activ33(x2)
        x2 = self.conv33(x2)
        x2 = self.relu33(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv41(x)
        x = self.bn41(x)
        x = self.relu41(x)
        x1 = self.conv42(x)
        x1 = self.bn42(x1)
        x1 = self.relu42(x1)
        x2 = self.bn43(x)
        x2 = self.activ43(x2)
        x2 = self.conv43(x2)
        x2 = self.relu43(x2)
        x = torch.cat([x1, x2], 1)

        x = self.maxpool5(x)

        x = self.conv61(x)
        x = self.bn61(x)
        x = self.relu61(x)
        x1 = self.conv62(x)
        x1 = self.bn62(x1)
        x1 = self.relu62(x1)
        x2 = self.bn63(x)
        x2 = self.activ63(x2)
        x2 = self.conv63(x2)
        x2 = self.relu63(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv71(x)
        x = self.bn71(x)
        x = self.relu71(x)
        x1 = self.conv72(x)
        x1 = self.bn72(x1)
        x1 = self.relu72(x1)
        x2 = self.bn73(x)
        x2 = self.activ73(x2)
        x2 = self.conv73(x2)
        x2 = self.relu73(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv81(x)
        x = self.bn81(x)
        x = self.relu81(x)
        x1 = self.conv82(x)
        x1 = self.bn82(x1)
        x1 = self.relu82(x1)
        x2 = self.bn83(x)
        x2 = self.activ83(x2)
        x2 = self.conv83(x2)
        x2 = self.relu83(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv91(x)
        x = self.bn91(x)
        x = self.relu91(x)
        x1 = self.conv92(x)
        x1 = self.bn92(x1)
        x1 = self.relu92(x1)
        x2 = self.bn93(x)
        x2 = self.activ93(x2)
        x2 = self.conv93(x2)
        x2 = self.relu93(x2)
        x = torch.cat([x1, x2], 1)

        x = self.maxpool10(x)

        x = self.conv111(x)
        x = self.bn111(x)
        x = self.relu111(x)
        x1 = self.conv112(x)
        x1 = self.bn112(x1)
        x1 = self.relu112(x1)
        x2 = self.conv113(x)
        x2 = self.bn113(x2)
        x2 = self.relu113(x2)
        x = torch.cat([x1, x2], 1)

        x = self.drop121(x)
        x = self.conv121(x)
        x = self.relu121(x)
        x = self.avgpool121(x)

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
