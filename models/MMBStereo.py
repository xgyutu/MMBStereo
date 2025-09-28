import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *
import math
import timm
from models.MMBA import MMBA




class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])
    

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)

        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]
    



class MMBStereo(nn.Module):
    def __init__(self, maxdisp=192, train_refine=False):
        super(MMBStereo, self).__init__()
        self.maxdisp = maxdisp
        self.feature = Feature()
        self.feature_up = FeatUp()
        self.train_refine = train_refine
        
        # self.EdgeGenerator = EdgeGenerator()
        self.edge_operator = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        scharr_x = torch.tensor([
                                [-47, 0, 47],
                                [-162, 0, 162],
                                [-47, 0, 47]
                            ], dtype=torch.float32) / 256.0
        self.edge_operator.weight.data = scharr_x.repeat(1, 3, 1, 1)
        self.edge_operator.weight.requires_grad = False

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
        )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1, 5, 5), padding=(0, 2, 2), stride=1)
        self.MMBA = MMBA()
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

        self.up_context = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
        )
        self.up_edge = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
        )
        

    def forward(self, left, right):
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)
        
        edge_fea = self.edge_operator(left)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp // 4)
        corr_volume = self.corr_stem(corr_volume)
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)
        volume = self.agg(feat_volume * corr_volume)
        cost, cost_context, cost_edge = self.MMBA(volume, features_left, edge_fea)

        
        size = cost.size()
        cost_context = self.up_context(cost_context)
        cost_edge = self.up_edge(cost_edge)
        
        
        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_samples = torch.arange(0, self.maxdisp // 4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp // 4, 1, 1).repeat(cost.shape[0], 1, cost.shape[3],
                                                                            cost.shape[4])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)
        pred_up = context_upsample(pred, spx_pred)
        
        disp_samples_edge = torch.arange(0, self.maxdisp // 4, dtype=cost_context.dtype, device=cost_context.device)
        disp_samples_edge = disp_samples_edge.view(1, self.maxdisp // 4, 1, 1).repeat(cost_context.shape[0], 1, cost_context.shape[3],
                                                                        cost_context.shape[4])
        pred_context = regression_topk(cost_context.squeeze(1), disp_samples_edge, 2)
        pred_edge = regression_topk(cost_edge.squeeze(1), disp_samples_edge, 2)
        pred_up_context = context_upsample(pred_context, spx_pred)
        pred_up_edge = context_upsample(pred_edge, spx_pred)
        

        if self.training:
            
            return [pred_up * 4, pred.squeeze(1) * 4, pred_up_context * 4, pred_up_edge * 4]

        else:
            return [pred_up * 4]


if __name__ == '__main__':
    model = MMBStereo().cuda()
    inputs1 = torch.randn(1, 3, 320, 1216).cuda()
    inputs2 = torch.randn(1, 3, 320, 1216).cuda()
    outputs = model(inputs1, inputs2)
    print("outputs:", outputs[0].shape)