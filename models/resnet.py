from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['resnet50', 'resnet101']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=False,
                    dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 适用于resnet-18和resnet-34 (group卷积不影响)
class BasicBlock(nn.Module):
    expansion: int = 1  # Block整体输出channel是输入channel的1倍 (18-layers和34-layers中是相同的)
    def __init__(self,
                in_channel: int,
                out_channel: int,
                stride: int = 1,
                downsample: Optional[nn.Module] = None,
                groups: int = 1,
                width_per_group: int = 64,
                dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        # 使用BN是不需要bias
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or width_per_group != 64:
            raise ValueError('BasicBlock only supports groups=1 and width_per_group=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        # [B, in_channel, H, W] -> [B, out_channel, H, W]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        '''
        若下采样为None, 对应实现residual结构, 可跳过
        若下采样为not None, 缩放对应维度的第一层(conv3_1,conv4_1,conv5_1),每层实施下采样, 得到捷径输出
        '''
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# 适用于resnet-50和resnet-101和resnet-152
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4  # Block整体输出channel是输入channel的4倍 (50-layers和101-layers和152-layers中相差4倍)

    def __init__(self,
                in_channel: int,
                out_channel: int,
                stride: int = 1,
                downsample: Optional[nn.Module] = None,
                groups: int = 1,
                width_per_group: int = 64,
                dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channel * (width_per_group / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channel, width)  # 压缩通道 squeeze channels
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 改为组卷积
        self.bn2 = norm_layer(width)
        # 无论是否group卷积，Block整体输出channel不变
        self.conv3 = conv1x1(width, out_channel * self.expansion)  # 解压通道 unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)  # 降维
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)  # 升维
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                block: Type[Union[BasicBlock, Bottleneck]],
                layers: List[int],
                num_classes: int = 1000,
                zero_init_residual: bool = False,
                groups: int = 1,
                width_per_group: int = 64,
                replace_stride_with_dilation: Optional[List[bool]] = None,
                norm_layer: Optional[Callable[..., nn.Module]] = None,
                args=None) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channel = 64  # max pool后得到特征矩阵的深度
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                            "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        # 全连接
        # [512*block.expansion, 1, 1] -> [num_classes, 1, 1]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    std_channel: int,  # std_channel表示此conv上每个block的第一个卷积层输出的卷积核个数
                    blocks: int,
                    stride: int = 1,
                    dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 确定下采样层
        # 对于resnet50,101,152的conv3,4,5(stride != 1)：需要调整特征矩阵的深度，且将高宽都缩减至一半
        if stride != 1 or self.in_channel != std_channel * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, std_channel * block.expansion, stride),
                norm_layer(std_channel * block.expansion),
            )

        layers = []
        # 构建第一层Block
        layers.append(block(self.in_channel,
                            std_channel, 
                            stride, 
                            downsample, 
                            self.groups, 
                            self.width_per_group, 
                            previous_dilation,
                            norm_layer))
        # 修改in_channel
        self.in_channel = std_channel * block.expansion
        # 从1开始，不需要下采样参数
        for _ in range(1, blocks):
            layers.append(block(self.in_channel,
                                std_channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group,
                                dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 非关键字参数传入

    def forward(self, x):
        # See note [TorchScript super()]
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

        x = self.avgpool(x)
        # 展平处理
        x = torch.flatten(x, 1)
        # [512*block.expansion, 1, 1] -> [num_classes, 1, 1]
        x = self.fc(x)
        return x

# 加入预训练模型
def _resnet(arch: str, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], pretrained: bool, progress: bool,
            **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(logger, args):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet50',
                    Bottleneck, [3, 4, 6, 3],
                    pretrained=args.pretrained,
                    progress=True,
                    num_classes=args.num_classes,
                    args=args)
    return model


def resnet101(logger, args):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet101',
                    Bottleneck, [3, 4, 23, 3],
                    pretrained=args.pretrained,
                    progress=True,
                    num_classes=args.num_classes,
                    args=args)
    return model
