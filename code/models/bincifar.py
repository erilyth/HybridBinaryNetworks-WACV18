import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.hardtanh = nn.Hardtanh()
        self.drop = nn.Dropout2d(0.2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)

        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding= 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding= 1)

        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

        self.bn5 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1)

        self.maxpool3 = nn.MaxPool2d(kernel_size = 2)

        self.bn7 = nn.BatchNorm2d(512)
        self.conv7 = nn.Conv2d(512, 1024, kernel_size = 4, stride = 1)
        #self.bn8 = nn.BatchNorm2d(1024,affine=False)
        #self.conv8 = BinConv2d(1024, 1024, kernel_size = 1, stride = 1)
        self.bn8 = nn.BatchNorm2d(1024)
        self.conv8 = nn.Conv2d(1024, 10, kernel_size = 1, stride = 1,  padding = 0)
        #self.bn10 = nn.BatchNorm2d(10,affine=False)


    def forward(self, x):

        #print(x.size())
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(self.relu(self.conv2(self.bn2(x))))

        x = self.relu(self.conv3(self.bn3(x)))
        x = self.maxpool2(self.relu(self.conv4(self.bn4(x))))
        #x = self.drop(x)
        #print(x.size())

        x = self.relu(self.conv5(self.bn5(x)))
        x = self.maxpool3(self.relu(self.conv6(self.bn6(x))))
        #x = self.drop(x)
        #print(x.size())
        x = self.relu(self.conv7(self.bn7(x)))
        #x = self.conv8(self.bn8(x))
        x = self.relu(self.conv8(self.bn8(x)))
        #print(x)
        x = x.view(-1, 10)

        return F.log_softmax(x)
