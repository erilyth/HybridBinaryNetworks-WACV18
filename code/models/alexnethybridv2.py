import torch.nn as nn
import torch.nn.functional as F
from newlayers.BinActiveZ import Active

class Net(nn.Module):
    def __init__(self,nClasses):
        super(Net, self).__init__()
        self.drop = nn.Dropout2d(0.0)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 2)
        self.bn1 = nn.BatchNorm2d(96)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        self.activ2 = Active()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding= 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.activ3 = Active()

        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding= 1, bias=False)
        self.bn4 = nn.BatchNorm2d(384)
        self.activ4 = Active()

        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding= 1, bias=False)
        self.bn5 = nn.BatchNorm2d(384)
        self.activ5 = Active()

        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv6 = nn.Conv2d(256, 4096, kernel_size = 6, stride = 1, padding = 0, bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.activ6 = Active()

        self.conv7 = nn.Conv2d(4096, 4096, kernel_size = 1, stride = 1, padding = 0, bias=False)
        self.bn7 = nn.BatchNorm2d(4096)

        self.conv8 = nn.Conv2d(4096, nClasses, kernel_size = 1, stride = 1)

        self.nClasses = nClasses


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(self.activ2(self.bn2(x))))
        x = self.maxpool2(x)
        x = self.relu(self.conv3(self.activ3(self.bn3(x))))
        x = self.relu(self.conv4(self.activ4(self.bn4(x))))
        x = self.relu(self.conv5(self.activ5(self.bn5(x))))
        x = self.maxpool3(x)
        x = self.drop(x)
        x = self.relu(self.conv6(self.activ6(self.bn6(x))))
        x = self.drop(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        x = x.view(-1, self.nClasses)

        return F.log_softmax(x)
