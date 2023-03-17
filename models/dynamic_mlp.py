import torch
import torch.nn as nn


# Residual MLP Backbone
class FCResLayer(nn.Module):
    def __init__(self, linear_size=256):
        super(FCResLayer, self).__init__()
        self.linear_size = linear_size
        self.linear1 = nn.Linear(self.linear_size, self.linear_size)
        self.linear2 = nn.Linear(self.linear_size, self.linear_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = self.relu(y)
        out = x + y
        return out

# FCNet(meta_net): 提取多模态特征
class FCNet(nn.Module):
    def __init__(self, num_inputs, num_classes=1000, num_filts=256):
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.Features = nn.Sequential(
            nn.Linear(num_inputs, num_filts),
            nn.ReLU(inplace=True),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
        )
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)

    def eval_single_class(self, x, class_of_interest):
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])

    def forward(self, x, return_feats=False, class_of_interest=None):
        # [bz, num_inputs] -> [bz, num_filts]
        loc_emb = self.Features(x)
        if return_feats:  # 只返回Feature
            return loc_emb  # [bz, num_filts]
        if class_of_interest is None:
            # [bz, num_filts] -> [bz, num_class]
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)
        return class_pred  # [bz, num_class]


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(conv, )
        if not bias:
            self.conv.add_module('ln', nn.LayerNorm(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


# basic implementation
class Dynamic_MLP_A(nn.Module):
    def __init__(self, in_channel, out_channel, meta_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.get_weight = nn.Linear(meta_channel, in_channel * out_channel)
        self.norm = nn.LayerNorm(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feature, meta_feature):
        ''' Reshape '''
        # [bz, meta_channel] -> [bz, in_channel*out_channel]
        weight = self.get_weight(meta_feature)
        # [bz, in_channel*out_channel] -> [bz, in_channel, out_channel]
        weight = weight.view(-1, self.in_channel, self.out_channel)
        ''' Matrix Multiplication '''
        # [bz, in_channel] -> [bz, 1, in_channel]
        img_feature = img_feature.unsqueeze(1)
        # [bz, 1, in_channel] * [bz, in_channel, out_channel] -> [bz, 1, out_channel]
        img_feature = torch.bmm(img_feature, weight)
        # [bz, 1, out_channel] -> [bz, out_channel]
        img_feature = img_feature.squeeze(1)

        img_feature = self.norm(img_feature)
        img_feature = self.relu(img_feature)
        return img_feature


# Dynamic_MLP_B 相比 Dynamic_MLP_A: with deeper embedding layers
class Dynamic_MLP_B(nn.Module):
    def __init__(self, in_channel, out_channel, meta_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv11 = Basic1d(in_channel, in_channel, True)
        self.conv12 = nn.Linear(in_channel, in_channel)
        self.conv21 = Basic1d(meta_channel, in_channel, True)
        self.conv22 = nn.Linear(in_channel, in_channel * out_channel)

        self.LN_ReLU = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(out_channel, out_channel, False)

    def forward(self, img_feature, meta_feature):
        ''' with deeper embedding layers '''
        # [bz, in_channel] -> [bz, in_channel] -> [bz, in_channel]
        img_feature = self.conv11(img_feature)
        img_feature = self.conv12(img_feature)
        # [bz, meta_channel] -> [bz, in_channel] -> [bz, in_channel*out_channel] -> [bz, in_channel, out_channel]
        meta_feature = self.conv21(meta_feature)
        meta_feature = self.conv22(meta_feature)
        meta_feature = meta_feature.view(-1, self.in_channel, self.out_channel)
        ''' Matrix Multiplication '''
        # [bz, in_channel] * [bz, in_channel, out_channel] -> [bz, out_channel]
        img_feature = torch.bmm(img_feature.unsqueeze(1), meta_feature).squeeze(1)
        img_feature = self.LN_ReLU(img_feature)
        # [bz, out_channel] -> [bz, out_channel]
        img_feature = self.conv3(img_feature)
        return img_feature


# Dynamic_MLP_C 相比 Dynamic_MLP_B: with concatenation inputs
class Dynamic_MLP_C(nn.Module):
    def __init__(self, in_channel, out_channel, meta_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv11 = Basic1d(in_channel + meta_channel, in_channel, True)
        self.conv12 = nn.Linear(in_channel, in_channel)
        self.conv21 = Basic1d(in_channel + meta_channel, in_channel, True)
        self.conv22 = nn.Linear(in_channel, in_channel * out_channel)

        self.LN_ReLU = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(out_channel, out_channel, False)

    def forward(self, img_feature, meta_feature):
        ''' with concatenation inputs '''
        # concat([bz, in_channel], [bz, meta_channel]) -> [bz, in_channel+meta_channel]
        cat_feature = torch.cat([img_feature, meta_feature], 1)
        ''' with deeper embedding layers '''
        # [bz, in_channel+meta_channel] -> [bz, in_channel] -> [bz, in_channel]
        cat_img_feature = self.conv11(cat_feature)
        cat_img_feature = self.conv12(cat_img_feature)
        # [bz, in_channel+meta_channel] -> [bz, in_channel] -> [bz, in_channel*out_channel] -> [bz, in_channel, out_channel]
        cat_meta_feature = self.conv21(cat_feature)
        cat_meta_feature = self.conv22(cat_meta_feature)
        cat_meta_feature = cat_meta_feature.view(-1, self.in_channel, self.out_channel)
        ''' Matrix Multiplication '''
        # [bz, in_channel] * [bz, in_channel, out_channel] -> [bz, out_channel]
        img_feature = torch.bmm(cat_img_feature.unsqueeze(1), cat_meta_feature).squeeze(1)
        img_feature = self.LN_ReLU(img_feature)
        # [bz, out_channel] -> [bz, out_channel]
        img_feature = self.conv3(img_feature)
        return img_feature



# 选择 Dynamic_MLP 类型
class RecursiveBlock(nn.Module):
    def __init__(self, in_channel, out_channel, meta_channel, mlp_type='c'):
        super().__init__()
        if mlp_type.lower() == 'a':
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == 'b':
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == 'c':
            MLP = Dynamic_MLP_C
        self.dynamic_conv = MLP(in_channel, out_channel, meta_channel)

    def forward(self, img_feature, meta_feature):
        img_feature = self.dynamic_conv(img_feature, meta_feature)
        # ！: 每块的 meta_feature 都返回原始 meta_feature
        return img_feature, meta_feature


# 融合模块：包含多层递归(since more dynamic projections can lead to better performance experimentally)
class FusionModule(nn.Module):
    def __init__(self, in_channel=2048, out_channel=256, hidden=64, num_layers=2, mlp_type='c'):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden = hidden

        self.conv1 = nn.Linear(in_channel, out_channel)  # 为image feature降维
        conv2 = []
        if num_layers == 1:  # 只有一层
            conv2.append(RecursiveBlock(out_channel, out_channel, meta_channel=out_channel, mlp_type=mlp_type))
        else:  # 多层
            conv2.append(RecursiveBlock(out_channel, hidden, meta_channel=out_channel, mlp_type=mlp_type))  # 第一层
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, meta_channel=out_channel, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, out_channel, meta_channel=out_channel, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)
        self.conv3 = nn.Linear(out_channel, in_channel)  # 为image feature升维
        self.norm3 = nn.LayerNorm(in_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_feature, meta_feature):
        '''
        img_feature: (N, channel), backbone输出经过全局池化的feature
        meta_feature: (N, fea_dim)
        '''
        identity = img_feature
        # 升维 Zi -> (Z^0)i
        img_feature = self.conv1(img_feature)
        for layer in self.conv2:
            # 递归DynamicMLP: (Z^n)i -> (Z^(n+1))i
            img_feature, meta_feature = layer(img_feature, meta_feature)
        # skip connection: 降维(Z^N)i -> Zi
        img_feature = self.conv3(img_feature)
        img_feature = self.norm3(img_feature)
        img_feature += identity
        return img_feature


# def get_dynamic_mlp(in_channel, args):
#     return FusionModule(in_channel=in_channel,
#                         out_channel=args.mlp_out_channel,
#                         hidden=args.mlp_hidden,
#                         num_layers=args.mlp_num_layers,
#                         mlp_type=args.mlp_type)
