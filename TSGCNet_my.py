import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


def knn(x, k):
    # use float16 to save GPU memory, HalfTensor is enough here; this reduced lots mem usage, so we can use 10GB GPU
    # (batch_size, dim, num_points)
    x = x.half()

    # bmm does not help on GPU memory here: inner = -2 * torch.bmm(x.transpose(2, 1), x)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)

    # use two steps for pairwise_distance, to save GPU memory; this reduced lots mem usage, so we can use 10GB GPU
    pairwise_distance = -xx - inner
    xx = xx.transpose(2, 1)
    pairwise_distance -= xx
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
    return idx

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class WeightedSumLocalFeatureAggregation(nn.Module):
    def __init__(self, feature_dim, out_dim, k):
        super(WeightedSumLocalFeatureAggregation, self).__init__()
        self.k = k

        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Conv2d(feature_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

        # 权重生成层
        self.weight_generation = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=1, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x 的形状预期为 [B, F, N, K]
        # 其中 F 是特征维度，N 是点数，K 是每个点的邻居数
        # 确保输入数据的形状符合卷积层的预期

        x = x.permute(0, 2, 3, 1)  # 转换 x 为 [B, N, K, F]

        # 确保特征转换层的输入形状是 [B, F, K, N]
        x = x.permute(0, 3, 2, 1)  # 转换 x 为 [B, F, K, N]

        transformed_features = self.feature_transform(x)  # [B, O, K, N]
        weights = self.weight_generation(x)  # [B, 1, K, N]

        # 使用加权和聚合特征
        aggregated_features = torch.sum(transformed_features * weights, dim=2)  # [B, O, N]

        return aggregated_features.permute(0, 1, 2)  # 转换为 [B, N, O]


def get_graph_feature(coor, nor, k=10):
    batch_size, num_dims, num_points  = coor.shape
    coor = coor.view(batch_size, -1, num_points)

    idx = knn(coor, k=k)
    index = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = coor.size()
    _, num_dims2, _ = nor.size()

    coor = coor.transpose(2,1).contiguous()
    nor = nor.transpose(2,1).contiguous()

    # coordinate
    coor_feature = coor.view(batch_size * num_points, -1)[idx, :]
    coor_feature = coor_feature.view(batch_size, num_points, k, num_dims)
    coor = coor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    coor_feature = torch.cat((coor_feature, coor), dim=3).permute(0, 3, 1, 2).contiguous()

    # normal vector
    nor_feature = nor.view(batch_size * num_points, -1)[idx, :]
    nor_feature = nor_feature.view(batch_size, num_points, k, num_dims2)
    nor = nor.view(batch_size, num_points, 1, num_dims2).repeat(1, 1, k, 1)
    nor_feature = torch.cat((nor_feature, nor), dim=3).permute(0, 3, 1, 2).contiguous()
    return coor_feature, nor_feature, index


class ModifiedWeightedSumLocalFeatureAggregation(nn.Module):
    def __init__(self, feature_dim, out_dim, K):
        super(ModifiedWeightedSumLocalFeatureAggregation, self).__init__()
        self.K = K

        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 权重生成层
        self.weight_generation = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 1, kernel_size=1, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, Graph_index, x, feature):
        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0, 2, 3, 1)
        neighbor_feature = index_points(x, Graph_index)
        center = x.view(B, N, 1, C).expand(B, N, self.K, C)

        # 组合中心点特征和邻居特征
        delta_feature = torch.cat([center - neighbor_feature, neighbor_feature], dim=3).permute(0, 3, 2, 1)

        # 特征转换
        transformed_features = self.feature_transform(delta_feature)  # [B, O, N, K]

        # 调整 feature 维度以匹配 transformed_features
        feature = feature.permute(0, 3, 2, 1)  # 确保 feature 的形状是 [B, N, K, F]

        # 权重生成
        weights = self.weight_generation(delta_feature)  # [B, 1, N, K]

        # 使用加权和聚合特征
        aggregated_features = torch.sum(transformed_features * weights * feature, dim=-1)  # [B, O, N]

        return aggregated_features.permute(0, 2, 1)  # [B, N, O]


class GraphAttention(nn.Module):
    def __init__(self,feature_dim,out_dim, K):
        super(GraphAttention, self).__init__()
        self.dropout = 0.6
        self.conv = nn.Sequential(nn.Conv2d(feature_dim * 2, out_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_dim),
                                     nn.LeakyReLU(negative_slope=0.2))
        self.K=K

    def forward(self, Graph_index, x, feature):

        B, C, N = x.shape
        x = x.contiguous().view(B, N, C)
        feature = feature.permute(0,2,3,1)
        neighbor_feature = index_points(x, Graph_index)
        centre = x.view(B, N, 1, C).expand(B, N, self.K, C)
        delta_f = torch.cat([centre-neighbor_feature, neighbor_feature], dim=3).permute(0,3,2,1)
        e = self.conv(delta_f)
        e = e.permute(0,3,2,1)
        attention = F.softmax(e, dim=2) # [B, npoint, nsample,D]
        graph_feature = torch.sum(torch.mul(attention, feature),dim = 2) .permute(0,2,1)
        return graph_feature

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.bn1 = nn.GroupNorm(1, in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.bn2 = nn.GroupNorm(1, in_planes // ratio)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.bn2(self.relu1(self.fc1(self.bn1(self.avg_pool(x))))))
        max_out = self.fc2(self.bn2(self.relu1(self.fc1(self.bn1(self.max_pool(x))))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = self.build_conv(channel)
        self.bn = nn.BatchNorm1d(channel)  # 添加BatchNorm层
    def build_conv(self, channel):
        k = self.calculate_kernel_size(channel)
        padding = (k - 1) // 2
        return nn.Conv1d(1, 1, kernel_size=k, padding=padding, bias=False)

    def calculate_kernel_size(self, channel):
        k = math.ceil(math.log2(channel) / self.gamma + self.b / self.gamma)
        k = k if k % 2 == 1 else k + 1
        return k

    def forward(self, x):
        b, c, l = x.size()
        y = self.avg_pool(x.view(b, c, l)).view(b, 1, c)
        y = self.conv(y)
        y = torch.sigmoid(y).view(b, c, 1)
        out = x * y.expand_as(x)
        return self.bn(out)

class nECA(nn.Module):
    def __init__(self, channel, gamma=1.5, b=1):
        super(nECA, self).__init__()
        self.gamma = gamma
        self.b = b
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 2D平均池化
        self.conv = self.build_conv(channel)
        self.bn = nn.BatchNorm2d(channel)  # 2D批处理规范化

    def build_conv(self, channel):
        k = self.calculate_kernel_size(channel)
        padding = (k - 1) // 2
        # 更改卷积层以接受 'channel' 输入通道并输出 'channel' 通道。
        return nn.Conv2d(channel, channel, kernel_size=(k, k), padding=(padding, padding), groups=channel, bias=False)

    def calculate_kernel_size(self, channel):
        k = math.ceil(math.log2(channel) / self.gamma + self.b / self.gamma)
        k = k if k % 2 == 1 else k + 1
        return k

    def forward(self, x):
        b, c, _, _ = x.size()  # 修改这里以处理4D张量
        y = self.avg_pool(x).view(b, c, 1, 1)  # 修改视图以匹配2D数据的维度
        y = self.conv(y)
        y = torch.sigmoid(y)
        out = x * y.expand_as(x)  # 应用注意力机制
        return self.bn(out)  # 应用批处理规范化



# class ECA(nn.Module):
#     def __init__(self, channel, gamma=2.0, b=1.0, use_dropout=False, dropout_prob=0.0):
#         super(ECA, self).__init__()
#         self.gamma = nn.Parameter(torch.tensor(gamma))
#         self.b = nn.Parameter(torch.tensor(b))
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.conv = self.build_conv(channel)
#         self.bn = nn.BatchNorm1d(channel)
#         self.use_dropout = use_dropout
#         if use_dropout:
#             self.dropout = nn.Dropout(dropout_prob)
#
#     def build_conv(self, channel):
#         k = self.calculate_kernel_size(channel)
#         padding = (k - 1) // 2
#         return nn.Conv1d(1, 1, kernel_size=k, padding=padding, bias=False)
#
#     def calculate_kernel_size(self, channel):
#         k = math.ceil(math.log2(channel) / self.gamma.item() + self.b.item() / self.gamma.item())
#         k = k if k % 2 == 1 else k + 1
#         return int(k)
#
#     def forward(self, x):
#         b, c, l = x.size()
#         y = self.avg_pool(x.view(b, c, l)).view(b, 1, c)
#         y = self.conv(y)
#         y = torch.sigmoid(y).view(b, c, 1)
#         out = x * y.expand_as(x)
#         if self.use_dropout:
#             out = self.dropout(out)
#         return self.bn(out)


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weights = torch.softmax(torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(x.size(-1)), dim=-1)
        return torch.bmm(weights, v)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        # 添加一个 1x1 的卷积来匹配通道数
        self.match_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                        bias=False) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.match_channels is not None:
            identity = self.match_channels(identity)


        out += identity
        return out


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.LeakyReLU(negative_slope=0.2))
        # 添加一个 1x1 的卷积来匹配通道数
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                        bias=False) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.match_channels is not None:
            identity = self.match_channels(identity)


        out += identity
        return out

class TSGCNet(nn.Module):
    def __init__(self, k=16, in_channels=12, output_channels=8):
        super(TSGCNet, self).__init__()
        self.k = k
        ''' coordinate stream '''
        # NOTE: we reduced to half size of original network;
        # we already use 4 points and 4 normals per face, that is enough input data, maybe it is too redundant, we could reduce data and increase network size
        out1 = 32 # in_channels*2 * 2
        out2 = out1 * 2
        out3 = out2 * 2
        out4 = out3 * 2
        self.bn1_c = nn.BatchNorm2d(out1)
        self.bn2_c = nn.BatchNorm2d(out2)
        self.bn3_c = nn.BatchNorm2d(out3)
        self.bn4_c = nn.BatchNorm1d(out4)
        # self.conv1_c = nn.Sequential(nn.Conv2d(in_channels*2, out1, kernel_size=1, bias=False),
        #                            self.bn1_c,
        #                            nn.LeakyReLU(negative_slope=0.2))
        #
        # self.conv2_c = nn.Sequential(nn.Conv2d(out1*2, out2, kernel_size=1, bias=False),
        #                            self.bn2_c,
        #                            nn.LeakyReLU(negative_slope=0.2))
        #
        # self.conv3_c = nn.Sequential(nn.Conv2d(out2*2, out3, kernel_size=1, bias=False),
        #                            self.bn3_c,
        #                            nn.LeakyReLU(negative_slope=0.2))

        self.conv1_c = ResidualBlock2(in_channels*2, out1)

        self.conv2_c = ResidualBlock2(out1*2, out2)

        self.conv3_c = ResidualBlock2(out2*2, out3)


        self.conv4_c = nn.Sequential(nn.Conv1d(out1+out2+out3, out4, kernel_size=1, bias=False),
                                     self.bn4_c,
                                     nn.LeakyReLU(negative_slope=0.2))

        # self.conv4_c = ResidualBlock (out1+out2+out3, out4)

        # SY 12 -> 3
        self.attention_layer1_c = GraphAttention(feature_dim=12, out_dim=out1, K=self.k)
        self.attention_layer2_c = GraphAttention(feature_dim=out1, out_dim=out2, K=self.k)
        self.attention_layer3_c = GraphAttention(feature_dim=out2, out_dim=out3, K=self.k)

        self.channel_attention_layer1 = ChannelAttention(out1)
        self.channel_attention_layer2 = ChannelAttention(out2)
        self.channel_attention_layer3 = ChannelAttention(out3)
        self.channel_attention_layer4 = ChannelAttention(out4*2)

        self.nECA1 = nECA(out1)
        self.nECA2 = nECA(out2)
        self.nECA3 = nECA(out3)

        # self.nECA1 = nECA(channel=64, gamma=2, b=1)
        # self.nECA2 = nECA(channel=128, gamma=1.5, b=1)
        # self.nECA3 = nECA(channel=256, gamma=1, b=1)

        self.eca1 = ECA(512)
        self.eca2 = ECA(256)
        self.eca3 = ECA(128)
        # SY 12 -> 3
        self.FTM_c1 = STNkd(k=12)
        ''' normal stream '''
        self.bn1_n = nn.BatchNorm2d(out1)
        self.bn2_n = nn.BatchNorm2d(out2)
        self.bn3_n = nn.BatchNorm2d(out3)
        self.bn4_n = nn.BatchNorm1d(out4)
        self.conv1_n = nn.Sequential(nn.Conv2d((in_channels)*2, out1, kernel_size=1, bias=False),
                                     self.bn1_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv2_n = nn.Sequential(nn.Conv2d(out1*2, out2, kernel_size=1, bias=False),
                                     self.bn2_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv3_n = nn.Sequential(nn.Conv2d(out2*2, out3, kernel_size=1, bias=False),
                                     self.bn3_n,
                                     nn.LeakyReLU(negative_slope=0.2))

        self.conv4_n = nn.Sequential(nn.Conv1d(out1+out2+out3, out4, kernel_size=1, bias=False),
                                     self.bn4_n,
                                     nn.LeakyReLU(negative_slope=0.2))
        # SY 12 -> 3
        self.FTM_n1 = STNkd(k=12)

        '''feature-wise attention'''

        self.fa = nn.Sequential(nn.Conv1d(out4*2, out4*2, kernel_size=1, bias=False),
                                nn.BatchNorm1d(out4*2),
                                nn.LeakyReLU(0.2))

        ''' feature fusion '''
        # self.pred1 = nn.Sequential(nn.Conv1d(out4*2, 512, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(512),
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.pred2 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(256),
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.pred3 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                            nn.BatchNorm1d(128),
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))
        #
        self.pred1 = ResidualBlock(out4 * 2, 512)
        self.pred2 = ResidualBlock(512, 256)
        self.pred3 = ResidualBlock(256, 128)
        self.pred4 = nn.Sequential(nn.Conv1d(128, output_channels, kernel_size=1, bias=False))

        # nn.Dropout(p): p, probability of an element to be zeroed
        # NOTE: original p=0.6, it may converge too slow, we change to smaller prob (as we dropout 3 times) so it will converge faster
        p = 0.03
        self.dp1 = nn.Dropout(p)
        self.dp2 = nn.Dropout(p)
        self.dp3 = nn.Dropout(p)

    def forward(self, x):
        # SY 12 -> 3
        coor = x[:, :12, :]
        # SY 12 -> 3
        nor = x[:, 12:, :]

        # transform
        trans_c = self.FTM_c1(coor)
        coor = coor.transpose(2, 1)
        coor = torch.bmm(coor, trans_c)
        coor = coor.transpose(2, 1)
        trans_n = self.FTM_n1(nor)
        nor = nor.transpose(2, 1)
        nor = torch.bmm(nor, trans_n)
        nor = nor.transpose(2, 1)

        coor1, nor1, index = get_graph_feature(coor, nor, k=self.k)
        coor1 = self.conv1_c(coor1)
        nor1 = self.conv1_n(nor1)
        coor1 = self.attention_layer1_c(index, coor, coor1)

        coor1 = self.channel_attention_layer1(coor1)
        # nor1 = self.nECA1(nor1)
        nor1 = nor1.max(dim=-1, keepdim=False)[0]

        del coor, nor


        coor2, nor2, index = get_graph_feature(coor1, nor1, k=self.k)
        coor2 = self.conv2_c(coor2)
        nor2 = self.conv2_n(nor2)
        coor2 = self.attention_layer2_c(index, coor1, coor2)
        coor2 = self.channel_attention_layer2(coor2)
        # nor2 = self.nECA2(nor2)

        nor2 = nor2.max(dim=-1, keepdim=False)[0]

        coor3, nor3, index = get_graph_feature(coor2, nor2, k=self.k)
        coor3 = self.conv3_c(coor3)
        nor3 = self.conv3_n(nor3)
        coor3 = self.attention_layer3_c(index, coor2, coor3)
        coor3 = self.channel_attention_layer3(coor3)
        # nor3 = self.nECA3(nor3)
        nor3 = nor3.max(dim=-1, keepdim=False)[0]

        coor = torch.cat((coor1, coor2, coor3), dim=1)
        coor = self.conv4_c(coor)
        nor = torch.cat((nor1, nor2, nor3), dim=1)
        nor = self.conv4_n(nor)
        del coor1, coor2, coor3, nor1, nor2, nor3

        avgSum_coor = coor.sum(1) / 512
        avgSum_nor = nor.sum(1) / 512
        avgSum = avgSum_coor + avgSum_nor
        weight_coor = (avgSum_coor / avgSum).unsqueeze(1)
        weight_nor = (avgSum_nor / avgSum).unsqueeze(1)
        x = torch.cat((coor * weight_coor, nor * weight_nor), dim=1)

        weight = self.fa(x)

        x = weight * x
        x = self.channel_attention_layer4(x)
        del coor, nor, weight


        x = self.pred1(x)
        x = self.eca1(x)
        self.dp1(x)
        x = self.pred2(x)
        self.dp2(x)
        x = self.eca2(x)
        x = self.pred3(x)
        self.dp3(x)
        x = self.eca3(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=1)
        score = score.permute(0, 2, 1)
        return score


if __name__ == "__main__":
    num_faces = 16000
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input size: [batch_size, C, N], where C is number of dimension, N is the number of mesh faces.
    x = torch.rand(1,24,num_faces)
    x = x.cuda()
    model = TSGCNet(in_channels=12, output_channels=17, k=32)
    model = model.cuda()
    y = model(x)
    print(y.shape)