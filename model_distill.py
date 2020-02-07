import torch
import torch.nn.functional as F
from torch import nn
from module.convGRU import ConvGRU
from module.convLSTM import ConvLSTM
from module.spa_module import STA2_Module
from matplotlib import pyplot as plt
from module.se_layer import SELayer
from resnext.resnext101 import ResNeXt101
from resnext.resnext50 import ResNeXt50
from resnext.resnet50 import ResNet50
from resnext.resnet101 import ResNet101
from functools import partial

class R3Net(nn.Module):
    def __init__(self, motion='GRU', se_layer=False, dilation=True, basic_model='resnext50'):
        super(R3Net, self).__init__()

        self.motion = motion
        self.se_layer = se_layer
        self.dilation = dilation
        if basic_model == 'resnext50':
            resnext = ResNeXt50()
        elif basic_model == 'resnext101':
            resnext = ResNeXt101()
        elif basic_model == 'resnet50':
            resnext = ResNet50()
        else:
            resnext = ResNet101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.reduce_low = nn.Sequential(
            nn.Conv2d(64 + 256 + 512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU()
        )
        self.reduce_high = nn.Sequential(
            nn.Conv2d(1024 + 2048, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU(),
            _ASPP(256)
        )

            # self.motion_predict = nn.Conv2d(256, 1, kernel_size=1)

        if self.se_layer:
            self.reduce_high_se = SELayer(256)
            self.reduce_low_se = SELayer(256)
            # self.motion_se = SELayer(32)

        if dilation:
            resnext.layer3.apply(partial(self._nostride_dilate, dilate=2))
            resnext.layer4.apply(partial(self._nostride_dilate, dilate=4))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        l0_size = layer0.size()[2:]
        reduce_low = self.reduce_low(torch.cat((
            layer0,
            F.upsample(layer1, size=l0_size, mode='bilinear', align_corners=True),
            F.upsample(layer2, size=l0_size, mode='bilinear', align_corners=True)), 1))
        reduce_high = self.reduce_high(torch.cat((
            layer3,
            F.upsample(layer4, size=layer3.size()[2:], mode='bilinear', align_corners=True)), 1))



        return reduce_high, reduce_low


class _ASPP(nn.Module):
    #  this module is proposed in deeplabv3 and we use it in all of our baselines
    def __init__(self, in_dim):
        super(_ASPP, self).__init__()
        down_dim = int(in_dim / 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',
                           align_corners=True)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))

class Distill(nn.Module):
    def __init__(self, basic_model='resnet50', seq=False, dilation=False):
        super(Distill, self).__init__()
        self.head = R3Net(motion='', se_layer=False, dilation=dilation, basic_model=basic_model)
        # self.relation = STA2_Module(512, 256)
        self.mutual_self = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.BatchNorm2d(256), nn.PReLU(),
                # nn.Conv2d(256, 256, kernel_size=1), nn.PReLU(),
        )
        self.predict0 = nn.Conv2d(256, 1, kernel_size=1)
        self.predict1 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predict2 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # self.predict3 = nn.Sequential(
        #     nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
        #     nn.Conv2d(128, 1, kernel_size=1)
        # )


        if seq:
            self.mutual = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.BatchNorm2d(256), nn.PReLU(),
                # nn.Conv2d(256, 256, kernel_size=1), nn.PReLU(),
            )
            # self.mutual_cur = nn.Sequential(
            #     nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.BatchNorm2d(256), nn.PReLU(),
            #     # nn.Conv2d(256, 256, kernel_size=1), nn.PReLU(),
            # )
            # self.mutual_next = nn.Sequential(
            #     nn.Conv2d(512, 256, kernel_size=1, padding=0), nn.BatchNorm2d(256), nn.PReLU(),
            #     # nn.Conv2d(256, 256, kernel_size=1), nn.PReLU(),
            # )
            

    def generate_attention(self, query, value, operation):
        b, c, h, w = query.size()
        value_a = value.view(b, c, h * w).permute(0, 2, 1)
        query_a = query.view(b, c, h * w)

        feat = torch.matmul(value_a, query_a)
        feat = F.softmax((c ** -.5) * feat, dim=-1)
        feat = torch.matmul(feat, query_a.permute(0, 2, 1)).permute(0, 2, 1)
        feat_mutual = torch.cat([feat, query_a], dim=1).view(b, 2 * c, h, w)

        feat_mutual = operation(feat_mutual)
        return feat_mutual

    def forward(self, pre, cur, next, flag='single'):
        up_size = (80, 80)
        if flag == 'single':
            feat_high_cur, feat_low_cur = self.head(cur)

            # feat_high_cur = self.relation(feat_high_cur, feat_high_cur)
            feat_high_cur = F.upsample(feat_high_cur, size=feat_low_cur.size()[2:], mode='bilinear', align_corners=True)
#             feat_low_cur = F.upsample(feat_low_cur, size=up_size, mode='bilinear', align_corners=True)
            
#             feat_cur = self.generate_attention(feat_high_cur, feat_low_cur, self.mutual_self)
            predict0 = self.predict0(feat_high_cur)
            predict1 = self.predict1(torch.cat((predict0, feat_low_cur), 1)) + predict0
            predict2 = self.predict2(torch.cat((predict1, feat_high_cur), 1)) + predict1
            # predict3 = self.predict3(torch.cat((predict2, feat_low_cur), 1)) + predict2

            predict0 = F.upsample(predict0, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict1 = F.upsample(predict1, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict2 = F.upsample(predict2, size=cur.size()[2:], mode='bilinear', align_corners=True)
            # predict3 = F.upsample(predict3, size=cur.size()[2:], mode='bilinear', align_corners=True)

            if self.training:
                return predict0, predict1, predict2
            else:
                return F.sigmoid(predict2)
        else:
            feat_high_pre, feat_low_pre = self.head(pre)
            feat_high_cur, feat_low_cur = self.head(cur)
            feat_high_next, feat_low_next = self.head(next)
            
            pre_feat = self.generate_attention(feat_high_pre, feat_high_cur, self.mutual)
            cur_feat = self.generate_attention(feat_high_cur, feat_high_pre, self.mutual) + \
                       self.generate_attention(feat_high_cur, feat_high_next, self.mutual)
            next_feat = self.generate_attention(feat_high_next, feat_high_cur,  self.mutual)
            
            pre_feat = F.upsample(pre_feat, size=feat_low_pre.size()[2:], mode='bilinear', align_corners=True)
            cur_feat = F.upsample(cur_feat, size=feat_low_cur.size()[2:], mode='bilinear', align_corners=True)
            next_feat = F.upsample(next_feat, size=feat_low_next.size()[2:], mode='bilinear', align_corners=True)
            
            feat_high_pre = F.upsample(feat_high_pre, size=feat_low_pre.size()[2:], mode='bilinear', align_corners=True)
            feat_high_cur = F.upsample(feat_high_cur, size=feat_low_cur.size()[2:], mode='bilinear', align_corners=True)
            feat_high_next = F.upsample(feat_high_next, size=feat_low_next.size()[2:], mode='bilinear', align_corners=True)
            
            predict0_pre = self.predict0(feat_high_pre)
            predict1_pre = self.predict1(torch.cat((predict0_pre, feat_low_pre), 1)) + predict0_pre
            predict2_pre = self.predict2(torch.cat((predict1_pre, pre_feat), 1)) + predict1_pre
            
            predict0_cur = self.predict0(feat_high_cur)
            predict1_cur = self.predict1(torch.cat((predict0_cur, feat_low_cur), 1)) + predict0_cur
            predict2_cur = self.predict2(torch.cat((predict1_cur, cur_feat), 1)) + predict1_cur
            
            predict0_next = self.predict0(feat_high_next)
            predict1_next = self.predict1(torch.cat((predict0_next, feat_low_next), 1)) + predict0_next
            predict2_next = self.predict2(torch.cat((predict1_next, next_feat), 1)) + predict1_next
            
            predict0_pre = F.upsample(predict0_pre, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict0_cur = F.upsample(predict0_cur, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict0_next = F.upsample(predict0_next, size=cur.size()[2:], mode='bilinear', align_corners=True)
            
            predict1_pre = F.upsample(predict1_pre, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict1_cur = F.upsample(predict1_cur, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict1_next = F.upsample(predict1_next, size=cur.size()[2:], mode='bilinear', align_corners=True)
            
            predict2_pre = F.upsample(predict2_pre, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict2_cur = F.upsample(predict2_cur, size=cur.size()[2:], mode='bilinear', align_corners=True)
            predict2_next = F.upsample(predict2_next, size=cur.size()[2:], mode='bilinear', align_corners=True)
            
            if self.training:
                return predict0_pre, predict0_cur, predict0_next, predict1_pre, predict1_cur, predict1_next, predict2_pre, predict2_cur, predict2_next
            else:
                return F.sigmoid(predict2_cur)
