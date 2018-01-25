import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from newlayers.BinActiveZ import Active


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module):

    def __init__(self, nClasses):
        super(ResNet, self).__init__()
        self.nClasses = nClasses

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn21 = nn.BatchNorm2d(64)
        self.activ21 = Active()
        self.conv21 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(64)
        self.relu22 = nn.ReLU(inplace=True)

        self.bn31 = nn.BatchNorm2d(64)
        self.activ31 = Active()
        self.conv31 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu31 = nn.ReLU(inplace=True)
        self.bn32 = nn.BatchNorm2d(64)
        self.activ32 = Active()
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.relu32 = nn.ReLU(inplace=True)

        self.bn41 = nn.BatchNorm2d(64)
        self.activ41 = Active()
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu41 = nn.ReLU(inplace=True)
        self.bn42 = nn.BatchNorm2d(128)
        self.activ42 = Active()
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.conv43 = nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False)
        self.bn43 = nn.BatchNorm2d(128)
        self.relu43= nn.ReLU(inplace=True)

        self.conv51 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn51 = nn.BatchNorm2d(128)
        self.relu51 = nn.ReLU(inplace=True)
        self.conv52 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn52 = nn.BatchNorm2d(128)
        self.relu52 = nn.ReLU(inplace=True)

        self.bn61 = nn.BatchNorm2d(128)
        self.activ61 = Active()
        self.conv61 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu61 = nn.ReLU(inplace=True)
        self.bn62 = nn.BatchNorm2d(256)
        self.activ62 = Active()
        self.conv62 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.conv63 = nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False)
        self.bn63 = nn.BatchNorm2d(256)
        self.relu63 = nn.ReLU(inplace=True)

        self.conv71 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn71 = nn.BatchNorm2d(256)
        self.relu71 = nn.ReLU(inplace=True)
        self.conv72 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn72 = nn.BatchNorm2d(256)
        self.relu72 = nn.ReLU(inplace=True)

        self.conv81 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn81 = nn.BatchNorm2d(512)
        self.relu81 = nn.ReLU(inplace=True)
        self.conv82 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn82 = nn.BatchNorm2d(512)
        self.conv83 = nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False)
        self.bn83 = nn.BatchNorm2d(512)
        self.relu83 = nn.ReLU(inplace=True)

        self.conv91 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn91 = nn.BatchNorm2d(512)
        self.relu91 = nn.ReLU(inplace=True)
        self.conv92 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn92 = nn.BatchNorm2d(512)
        self.relu92 = nn.ReLU(inplace=True)

        self.avgpool101 = nn.AvgPool2d(7)
        self.linear111 = nn.Conv2d(512, nClasses, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)


        residual = x
        out = self.bn21(x)
        out = self.activ21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.conv22(out)
        out = self.bn22(out)
        out += residual
        out = self.relu22(out)

        residual = out
        out = self.bn31(x)
        out = self.activ31(out)
        out = self.conv31(out)
        out = self.relu31(out)
        out = self.bn32(out)
        out = self.activ32(out)
        out = self.conv32(out)
        out += residual
        out = self.relu32(out)


        residual = out
        out = self.bn41(out)
        out = self.activ41(out)
        out = self.conv41(out)
        out = self.relu41(out)
        out = self.bn42(out)
        out = self.activ42(out)
        out = self.conv42(out)
        residual = self.conv43(residual)
        residual = self.bn43(residual)
        out += residual
        out = self.relu43(out)

        residual = out
        out = self.conv51(out)
        out = self.bn51(out)
        out = self.relu51(out)
        out = self.conv52(out)
        out = self.bn52(out)
        out += residual
        out = self.relu52(out)


        residual = out
        out = self.bn61(out)
        out = self.activ61(out)
        out = self.conv61(out)
        out = self.relu61(out)
        out = self.bn62(out)
        out = self.activ62(out)
        out = self.conv62(out)
        residual = self.conv63(residual)
        residual = self.bn63(residual)
        out += residual
        out = self.relu63(out)

        residual = out
        out = self.conv71(out)
        out = self.bn71(out)
        out = self.relu71(out)
        out = self.conv72(out)
        out = self.bn72(out)
        out += residual
        out = self.relu72(out)


        residual = out
        out = self.conv81(out)
        out = self.bn81(out)
        out = self.relu81(out)
        out = self.conv82(out)
        out = self.bn82(out)
        residual = self.conv83(residual)
        residual = self.bn83(residual)
        out += residual
        out = self.relu83(out)

        residual = out
        out = self.conv91(out)
        out = self.bn91(out)
        out = self.relu91(out)
        out = self.conv92(out)
        out = self.bn92(out)
        out += residual
        out = self.relu92(out)

        out = self.avgpool101(out)
        x = self.linear111(out)

        x = x.view(-1, self.nClasses)

        return F.log_softmax(x)


def resnet18(nClasses, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(nClasses)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
