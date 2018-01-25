import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,nClasses):
        super(Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        #self.hardtanh = nn.Hardtanh()
        self.drop = nn.Dropout2d(0.5)

        self.conv1 = nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 2)
        self.bn1 = nn.BatchNorm2d(96)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2,  bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding= 1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding= 1, bias=False)
        self.bn4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding= 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv6 = nn.Conv2d(256, 4096, kernel_size = 6, stride = 1, padding = 0, bias=False)
        self.bn6 = nn.BatchNorm2d(4096)

        self.conv7 = nn.Conv2d(4096, 4096, kernel_size = 1, stride = 1, padding = 0, bias=False)
        self.bn7 = nn.BatchNorm2d(4096)

        self.conv8 = nn.Conv2d(4096, nClasses, kernel_size = 1, stride = 1)

        self.nClasses = nClasses


    def forward(self, x):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.maxpool3(x)
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.drop(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.drop(x)
        x = self.conv8(x)
        x = x.view(-1, self.nClasses)

        return F.log_softmax(x)
