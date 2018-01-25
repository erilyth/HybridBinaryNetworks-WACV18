import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, nClasses):
        super(Net, self).__init__()

        #pre layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size = 3, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.mp2 = nn.MaxPool2d(3, stride=2, padding=1)

        #a3 block
        in_planes_1, n1x1_1, n3x3red_1, n3x3_1, n5x5red_1, n5x5_1, pool_planes_1 = 192,  64,  96, 128, 16, 32, 32

        self.conv_1_1 = nn.Conv2d(in_planes_1, n1x1_1, kernel_size=1)
        self.bn_1_1 = nn.BatchNorm2d(n1x1_1)
        self.relu = nn.ReLU()

        self.conv_1_2 = nn.Conv2d(in_planes_1, n3x3red_1, kernel_size=1)
        self.bn_1_2 = nn.BatchNorm2d(n3x3red_1)
        self.relu = nn.ReLU()
        self.conv_1_3 = nn.Conv2d(n3x3red_1, n3x3_1, kernel_size=3, padding=1)
        self.bn_1_3 = nn.BatchNorm2d(n3x3_1)
        self.relu = nn.ReLU()

        self.conv_1_4 = nn.Conv2d(in_planes_1, n5x5red_1, kernel_size=1)
        self.bn_1_4 = nn.BatchNorm2d(n5x5red_1)
        self.relu = nn.ReLU()
        self.conv_1_5 = nn.Conv2d(n5x5red_1, n5x5_1, kernel_size=5, padding=2)
        self.bn_1_5 = nn.BatchNorm2d(n5x5_1)
        self.relu = nn.ReLU()

        self.mp_1_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_1_6 = nn.Conv2d(in_planes_1, pool_planes_1, kernel_size=1)
        self.bn_1_6 = nn.BatchNorm2d(pool_planes_1)
        self.relu = nn.ReLU()

        #b3 block
        in_planes_2, n1x1_2, n3x3red_2, n3x3_2, n5x5red_2, n5x5_2, pool_planes_2 = 256, 128, 128, 192, 32, 96, 64

        self.conv_2_1 = nn.Conv2d(in_planes_2, n1x1_2, kernel_size=1)
        self.bn_2_1 = nn.BatchNorm2d(n1x1_2)
        self.relu = nn.ReLU()

        self.conv_2_2 = nn.Conv2d(in_planes_2, n3x3red_2, kernel_size=1)
        self.bn_2_2 = nn.BatchNorm2d(n3x3red_2)
        self.relu = nn.ReLU()
        self.conv_2_3 = nn.Conv2d(n3x3red_2, n3x3_2, kernel_size=3, padding=1)
        self.bn_2_3 = nn.BatchNorm2d(n3x3_2)
        self.relu = nn.ReLU()

        self.conv_2_4 = nn.Conv2d(in_planes_2, n5x5red_2, kernel_size=1)
        self.bn_2_4 = nn.BatchNorm2d(n5x5red_2)
        self.relu = nn.ReLU()
        self.conv_2_5 = nn.Conv2d(n5x5red_2, n5x5_2, kernel_size=5, padding=2)
        self.bn_2_5 = nn.BatchNorm2d(n5x5_2)
        self.relu = nn.ReLU()

        self.mp_2_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_2_6 = nn.Conv2d(in_planes_2, pool_planes_2, kernel_size=1)
        self.bn_2_6 = nn.BatchNorm2d(pool_planes_2)
        self.relu = nn.ReLU()

        self.mp3 = nn.MaxPool2d(3, stride=2)

        #a4 block
        in_planes_3, n1x1_3, n3x3red_3, n3x3_3, n5x5red_3, n5x5_3, pool_planes_3 = 480, 192,  96, 204, 16,  48,  64

        self.conv_3_1 = nn.Conv2d(in_planes_3, n1x1_3, kernel_size=1)
        self.bn_3_1 = nn.BatchNorm2d(n1x1_3)
        self.relu = nn.ReLU()

        self.conv_3_2 = nn.Conv2d(in_planes_3, n3x3red_3, kernel_size=1)
        self.bn_3_2 = nn.BatchNorm2d(n3x3red_3)
        self.relu = nn.ReLU()
        self.conv_3_3 = nn.Conv2d(n3x3red_3, n3x3_3, kernel_size=3, padding=1)
        self.bn_3_3 = nn.BatchNorm2d(n3x3_3)
        self.relu = nn.ReLU()

        self.conv_3_4 = nn.Conv2d(in_planes_3, n5x5red_3, kernel_size=1)
        self.bn_3_4 = nn.BatchNorm2d(n5x5red_3)
        self.relu = nn.ReLU()
        self.conv_3_5 = nn.Conv2d(n5x5red_3, n5x5_3, kernel_size=5, padding=2)
        self.bn_3_5 = nn.BatchNorm2d(n5x5_3)
        self.relu = nn.ReLU()

        self.mp_3_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_3_6 = nn.Conv2d(in_planes_3, pool_planes_3, kernel_size=1)
        self.bn_3_6 = nn.BatchNorm2d(pool_planes_3)
        self.relu = nn.ReLU()

        #b4 block
        in_planes_4, n1x1_4, n3x3red_4, n3x3_4, n5x5red_4, n5x5_4, pool_planes_4 = 508, 160, 112, 224, 24,  64,  64

        self.conv_4_1 = nn.Conv2d(in_planes_4, n1x1_4, kernel_size=1)
        self.bn_4_1 = nn.BatchNorm2d(n1x1_4)
        self.relu = nn.ReLU()

        self.conv_4_2 = nn.Conv2d(in_planes_4, n3x3red_4, kernel_size=1)
        self.bn_4_2 = nn.BatchNorm2d(n3x3red_4)
        self.relu = nn.ReLU()
        self.conv_4_3 = nn.Conv2d(n3x3red_4, n3x3_4, kernel_size=3, padding=1)
        self.bn_4_3 = nn.BatchNorm2d(n3x3_4)
        self.relu = nn.ReLU()

        self.conv_4_4 = nn.Conv2d(in_planes_4, n5x5red_4, kernel_size=1)
        self.bn_4_4 = nn.BatchNorm2d(n5x5red_4)
        self.relu = nn.ReLU()
        self.conv_4_5 = nn.Conv2d(n5x5red_4, n5x5_4, kernel_size=5, padding=2)
        self.bn_4_5 = nn.BatchNorm2d(n5x5_4)
        self.relu = nn.ReLU()

        self.mp_4_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_4_6 = nn.Conv2d(in_planes_4, pool_planes_4, kernel_size=1)
        self.bn_4_6 = nn.BatchNorm2d(pool_planes_4)
        self.relu = nn.ReLU()

        #c4 block
        in_planes_5, n1x1_5, n3x3red_5, n3x3_5, n5x5red_5, n5x5_5, pool_planes_5 = 512, 128, 128, 256, 24,  64,  64

        self.conv_5_1 = nn.Conv2d(in_planes_5, n1x1_5, kernel_size=1)
        self.bn_5_1 = nn.BatchNorm2d(n1x1_5)
        self.relu = nn.ReLU()

        self.conv_5_2 = nn.Conv2d(in_planes_5, n3x3red_5, kernel_size=1)
        self.bn_5_2 = nn.BatchNorm2d(n3x3red_5)
        self.relu = nn.ReLU()
        self.conv_5_3 = nn.Conv2d(n3x3red_5, n3x3_5, kernel_size=3, padding=1)
        self.bn_5_3 = nn.BatchNorm2d(n3x3_5)
        self.relu = nn.ReLU()

        self.conv_5_4 = nn.Conv2d(in_planes_5, n5x5red_5, kernel_size=1)
        self.bn_5_4 = nn.BatchNorm2d(n5x5red_5)
        self.relu = nn.ReLU()
        self.conv_5_5 = nn.Conv2d(n5x5red_5, n5x5_5, kernel_size=5, padding=2)
        self.bn_5_5 = nn.BatchNorm2d(n5x5_5)
        self.relu = nn.ReLU()

        self.mp_5_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_5_6 = nn.Conv2d(in_planes_5, pool_planes_5, kernel_size=1)
        self.bn_5_6 = nn.BatchNorm2d(pool_planes_5)
        self.relu = nn.ReLU()

        #d4 block
        in_planes_6, n1x1_6, n3x3red_6, n3x3_6, n5x5red_6, n5x5_6, pool_planes_6 = 512, 112, 144, 288, 32,  64,  64

        self.conv_6_1 = nn.Conv2d(in_planes_6, n1x1_6, kernel_size=1)
        self.bn_6_1 = nn.BatchNorm2d(n1x1_6)
        self.relu = nn.ReLU()

        self.conv_6_2 = nn.Conv2d(in_planes_6, n3x3red_6, kernel_size=1)
        self.bn_6_2 = nn.BatchNorm2d(n3x3red_6)
        self.relu = nn.ReLU()
        self.conv_6_3 = nn.Conv2d(n3x3red_6, n3x3_6, kernel_size=3, padding=1)
        self.bn_6_3 = nn.BatchNorm2d(n3x3_6)
        self.relu = nn.ReLU()

        self.conv_6_4 = nn.Conv2d(in_planes_6, n5x5red_6, kernel_size=1)
        self.bn_6_4 = nn.BatchNorm2d(n5x5red_6)
        self.relu = nn.ReLU()
        self.conv_6_5 = nn.Conv2d(n5x5red_6, n5x5_6, kernel_size=5, padding=2)
        self.bn_6_5 = nn.BatchNorm2d(n5x5_6)
        self.relu = nn.ReLU()

        self.mp_6_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_6_6 = nn.Conv2d(in_planes_6, pool_planes_6, kernel_size=1)
        self.bn_6_6 = nn.BatchNorm2d(pool_planes_6)
        self.relu = nn.ReLU()

        #e4 block
        in_planes_7, n1x1_7, n3x3red_7, n3x3_7, n5x5red_7, n5x5_7, pool_planes_7 = 528, 256, 160, 320, 32, 128, 128

        self.conv_7_1 = nn.Conv2d(in_planes_7, n1x1_7, kernel_size=1)
        self.bn_7_1 = nn.BatchNorm2d(n1x1_7)
        self.relu = nn.ReLU()

        self.conv_7_2 = nn.Conv2d(in_planes_7, n3x3red_7, kernel_size=1)
        self.bn_7_2 = nn.BatchNorm2d(n3x3red_7)
        self.relu = nn.ReLU()
        self.conv_7_3 = nn.Conv2d(n3x3red_7, n3x3_7, kernel_size=3, padding=1)
        self.bn_7_3 = nn.BatchNorm2d(n3x3_7)
        self.relu = nn.ReLU()

        self.conv_7_4 = nn.Conv2d(in_planes_7, n5x5red_7, kernel_size=1)
        self.bn_7_4 = nn.BatchNorm2d(n5x5red_7)
        self.relu = nn.ReLU()
        self.conv_7_5 = nn.Conv2d(n5x5red_7, n5x5_7, kernel_size=5, padding=2)
        self.bn_7_5 = nn.BatchNorm2d(n5x5_7)
        self.relu = nn.ReLU()

        self.mp_7_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_7_6 = nn.Conv2d(in_planes_7, pool_planes_7, kernel_size=1)
        self.bn_7_6 = nn.BatchNorm2d(pool_planes_7)
        self.relu = nn.ReLU()

        self.mp4 = nn.MaxPool2d(3, stride=2, padding=1)

        #a5 block
        in_planes_8, n1x1_8, n3x3red_8, n3x3_8, n5x5red_8, n5x5_8, pool_planes_8 = 832, 256, 160, 320, 48, 128, 128

        self.conv_8_1 = nn.Conv2d(in_planes_8, n1x1_8, kernel_size=1)
        self.bn_8_1 = nn.BatchNorm2d(n1x1_8)
        self.relu = nn.ReLU()

        self.conv_8_2 = nn.Conv2d(in_planes_8, n3x3red_8, kernel_size=1)
        self.bn_8_2 = nn.BatchNorm2d(n3x3red_8)
        self.relu = nn.ReLU()
        self.conv_8_3 = nn.Conv2d(n3x3red_8, n3x3_8, kernel_size=3, padding=1)
        self.bn_8_3 = nn.BatchNorm2d(n3x3_8)
        self.relu = nn.ReLU()

        self.conv_8_4 = nn.Conv2d(in_planes_8, n5x5red_8, kernel_size=1)
        self.bn_8_4 = nn.BatchNorm2d(n5x5red_8)
        self.relu = nn.ReLU()
        self.conv_8_5 = nn.Conv2d(n5x5red_8, n5x5_8, kernel_size=5, padding=2)
        self.bn_8_5 = nn.BatchNorm2d(n5x5_8)
        self.relu = nn.ReLU()

        self.mp_8_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_8_6 = nn.Conv2d(in_planes_8, pool_planes_8, kernel_size=1)
        self.bn_8_6 = nn.BatchNorm2d(pool_planes_8)
        self.relu = nn.ReLU()

        #b5 block
        in_planes_9, n1x1_9, n3x3red_9, n3x3_9, n5x5red_9, n5x5_9, pool_planes_9 = 832, 384, 192, 384, 48, 128, 128

        self.conv_9_1 = nn.Conv2d(in_planes_9, n1x1_9, kernel_size=1)
        self.bn_9_1 = nn.BatchNorm2d(n1x1_9)
        self.relu = nn.ReLU()

        self.conv_9_2 = nn.Conv2d(in_planes_9, n3x3red_9, kernel_size=1)
        self.bn_9_2 = nn.BatchNorm2d(n3x3red_9)
        self.relu = nn.ReLU()
        self.conv_9_3 = nn.Conv2d(n3x3red_9, n3x3_9, kernel_size=3, padding=1)
        self.bn_9_3 = nn.BatchNorm2d(n3x3_9)
        self.relu = nn.ReLU()

        self.conv_9_4 = nn.Conv2d(in_planes_9, n5x5red_9, kernel_size=1)
        self.bn_9_4 = nn.BatchNorm2d(n5x5red_9)
        self.relu = nn.ReLU()
        self.conv_9_5 = nn.Conv2d(n5x5red_9, n5x5_9, kernel_size=5, padding=2)
        self.bn_9_5 = nn.BatchNorm2d(n5x5_9)
        self.relu = nn.ReLU()

        self.mp_9_6 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_9_6 = nn.Conv2d(in_planes_9, pool_planes_9, kernel_size=1)
        self.bn_9_6 = nn.BatchNorm2d(pool_planes_9)
        self.relu = nn.ReLU()

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, nClasses)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp2(x)

        #a3
        x1 = self.conv_1_1(x)
        x1 = self.bn_1_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_1_2(x)
        x2 = self.bn_1_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_1_3(x2)
        x2 = self.bn_1_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_1_4(x)
        x3 = self.bn_1_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_1_5(x3)
        x3 = self.bn_1_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_1_6(x)
        x4 = self.conv_1_6(x4)
        x4 = self.bn_1_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)

        #b3

        x1 = self.conv_2_1(x)
        x1 = self.bn_2_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_2_2(x)
        x2 = self.bn_2_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_2_3(x2)
        x2 = self.bn_2_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_2_4(x)
        x3 = self.bn_2_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_2_5(x3)
        x3 = self.bn_2_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_2_6(x)
        x4 = self.conv_2_6(x4)
        x4 = self.bn_2_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)

        x = self.mp3(x)
        #a4

        x1 = self.conv_3_1(x)
        x1 = self.bn_3_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_3_2(x)
        x2 = self.bn_3_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_3_3(x2)
        x2 = self.bn_3_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_3_4(x3)
        x3 = self.bn_3_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_3_5(x3)
        x3 = self.bn_3_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_3_6(x)
        x4 = self.conv_3_6(x4)
        x4 = self.bn_3_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        #b4

        x1 = self.conv_4_1(x)
        x1 = self.bn_4_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_4_2(x)
        x2 = self.bn_4_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_4_3(x2)
        x2 = self.bn_4_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_4_4(x)
        x3 = self.bn_4_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_4_5(x3)
        x3 = self.bn_4_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_4_6(x)
        x4 = self.conv_4_6(x4)
        x4 = self.bn_4_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        #c4

        x1 = self.conv_5_1(x)
        x1 = self.bn_5_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_5_2(x)
        x2 = self.bn_5_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_5_3(x2)
        x2 = self.bn_5_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_5_4(x)
        x3 = self.bn_5_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_5_5(x3)
        x3 = self.bn_5_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_5_6(x4)
        x4 = self.conv_5_6(x4)
        x4 = self.bn_5_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        #d4

        x1 = self.conv_6_1(x)
        x1 = self.bn_6_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_6_2(x)
        x2 = self.bn_6_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_6_3(x2)
        x2 = self.bn_6_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_6_4(x)
        x3 = self.bn_6_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_6_5(x3)
        x3 = self.bn_6_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_6_6(x)
        x4 = self.conv_6_6(x4)
        x4 = self.bn_6_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        #e4

        x1 = self.conv_7_1(x)
        x1 = self.bn_7_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_7_2(x)
        x2 = self.bn_7_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_7_3(x2)
        x2 = self.bn_7_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_7_4(x)
        x3 = self.bn_7_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_7_5(x3)
        x3 = self.bn_7_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_7_6(x)
        x4 = self.conv_7_6(x4)
        x4 = self.bn_7_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)

        x = self.mp4(x)
        #a5

        x1 = self.conv_8_1(x)
        x1 = self.bn_8_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_8_2(x)
        x2 = self.bn_8_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_8_3(x2)
        x2 = self.bn_8_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_8_4(x)
        x3 = self.bn_8_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_8_5(x3)
        x3 = self.bn_8_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_8_6(x)
        x4 = self.conv_8_6(x4)
        x4 = self.bn_8_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        #b5

        x1 = self.conv_9_1(x)
        x1 = self.bn_9_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv_9_2(x)
        x2 = self.bn_9_2(x2)
        x2 = self.relu(x2)
        x2 = self.conv_9_3(x2)
        x2 = self.bn_9_3(x2)
        x2 = self.relu(x2)

        x3 = self.conv_9_4(x)
        x3 = self.bn_9_4(x3)
        x3 = self.relu(x3)
        x3 = self.conv_9_5(x3)
        x3 = self.bn_9_5(x3)
        x3 = self.relu(x3)

        x4 = self.mp_9_6(x)
        x4 = self.conv_9_6(x4)
        x4 = self.bn_9_6(x4)
        x4 = self.relu(x4)

        x = torch.cat([x1, x2, x3, x4], 1)

        x = self.avgpool(x)
        x = self.linear(x)

        return x
