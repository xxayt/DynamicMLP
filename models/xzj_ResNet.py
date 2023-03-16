import torch.nn as nn
import torch

# 适用于resnet-18和resnet-34
class BasicBlock(nn.Module):
    expansion = 1  # Block整体输出channel是输入channel的1倍 (18-layers和34-layers中是相同的)

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # stride==1时,out_channel=in_channel; stride==2时,out_channel=in_channel/2
        # stride==2时，out_channel=in_channel/2
        # 使用BN是不需要bias
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        '''
        若下采样为None, 对应实现residual结构, 可跳过
        若下采样为not None, 缩放对应维度的第一层(conv3_1,conv4_1,conv5_1),每层实施下采样, 得到捷径输出
        '''
        if self.downsample is not None:
            # [] -> []
            identity = self.downsample(x)
        # [B, in_channel, H, W] -> [B, out_channel, H, W]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


# 适用于resnet-50和resnet-101和resnet-152
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # Block整体输出channel是输入channel的4倍 (50-layers和101-layers和152-layers中相差4倍)

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)  # 压缩通道 squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)  # 解压通道 unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)  # 降维
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) 

        out = self.conv3(out)  # 升维
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                block, 
                blocks_num, 
                num_classes=1000, 
                include_top=True):  # 为了在resnet基础上搭建更复杂网络，此处include_top不使用，但进行了实现
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # max pool后得到特征矩阵的深度

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0], stride=1)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            '''
            自适应的平均池化下采样操作：
            (h,w)=(1,1)相当于全局平均池化(Global Average Pooling, GAP)，代替全连接层，但参数量大幅减少
            [512*block.expansion, 7, 7] -> [512*block.expansion, 1, 1]
            '''
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 全连接
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self,
                    block, 
                    std_channel,  # std_channel表示此conv上每个block的第一个卷积层输出的卷积核个数
                    block_num, 
                    stride):
        DownSample = None
        # 确定下采样层
        # 对于resnet50,101,152的conv3,4,5(stride != 1)：需要调整特征矩阵的深度，且将高宽都缩减至一半
        if stride != 1 or self.in_channel != std_channel * block.expansion:
            DownSample = nn.Sequential(
                nn.Conv2d(self.in_channel, std_channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(std_channel * block.expansion))
        layers = []
        # 构建第一层Block
        layers.append(block(self.in_channel,
                            std_channel,
                            downsample=DownSample,
                            stride=stride))
        # 修改in_channel
        self.in_channel = std_channel * block.expansion
        # 从1开始，不需要下采样参数
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, std_channel))
        return nn.Sequential(*layers)  # 非关键字参数传入

    def forward(self, x):
        # [3,224,224] -> [64,112,112]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # [64,112,112] -> [64,56,56]
        x = self.maxpool(x)

        # [64, 56, 56] -> [64*block.expansion, 56, 56]
        x = self.layer1(x)
        # [64*block.expansion, 56, 56] -> [128*block.expansion, 28, 28]
        x = self.layer2(x)
        # [128*block.expansion, 28, 28] -> [256*block.expansion, 14, 14]
        x = self.layer3(x)
        # [256*block.expansion, 14, 14] -> [512*block.expansion, 7, 7]
        x = self.layer4(x)
        
        if self.include_top:
            x = self.avgpool(x)
            # 展平处理
            x = torch.flatten(x, 1)
            # [512*block.expansion, 1, 1] -> [num_classes, 1, 1]
            x = self.fc(x)
        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)