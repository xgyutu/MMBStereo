import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from mamba_ssm import Mamba

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x

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

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x
    
class Block(SubModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 3,
                 kernel_size: int = 5,
                 do_res: int = True,
                 n_groups: int or None = None,

                 ):
        super().__init__()

        self.do_res = do_res

        # First convolution layer with DepthWise Convolutions

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.LeakyReLU(True)

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.weight_init()

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1
    
    
class mambaBlock(SubModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 3,
                 kernel_size: int = 5,
                 do_res: int = True,
                 n_groups: int or None = None,
                 d_state = 16,
                 d_conv = 4,
                 expand = 2

                 ):
        super().__init__()

        self.do_res = do_res
        self.dim = in_channels

        # First convolution layer with DepthWise Convolutions
        self.norm1 = nn.LayerNorm(in_channels)
        self.mamba = Mamba(
                d_model=in_channels, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
        )

        self.norm2 = LayerNorm(in_channels)

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # GeLU activations
        self.act = nn.ReLU(True)

        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = nn.Conv3d(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        self.weight_init()
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, dummy_tensor=None):
        res = x
        
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm1(x_flat)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        res = res + x
        
        x = self.act(self.conv2(self.norm2(x)))
        x = self.conv3(x)
        x = res + x
        return x
    
    
    
class BasicBlock3d(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock3d, self).__init__()
        if inplanes == planes:
            self.if_res = True
        else:
            self.if_res = False
            
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes, momentum=0.1)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.if_res:
            out += residual

        return self.relu(out)
    

class dec_Block(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 exp_r: int = 4,
                 kernel_size: int = 3,
                 do_res: int = True,
                 n_groups: int or None = None,

                 ):
        super().__init__()

        self.do_res = do_res

        # First convolution layer with DepthWise Convolutions

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=3 // 2,
            groups=in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        self.norm = nn.GroupNorm(
            num_groups=in_channels,
            num_channels=in_channels
        )
        # GeLU activations
        self.act = nn.LeakyReLU(inplace=True)


    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.act(self.conv(self.norm(x1)))
        if self.do_res:
            x1 = x + x1
        return x1

class DownBlock(Block):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False):

        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)

        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class UpBlock(Block):

    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7,
                 do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size,
                         do_res=False)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2
            )

        self.conv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
        )

    def forward(self, x, dummy_tensor=None):

        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        x1 = torch.nn.functional.pad(x1, (1, 0, 1, 0, 1, 0))

        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, (1, 0, 1, 0, 1, 0))
            x1 = x1 + res

        return x1
        
        
class context_fusion(nn.Module):
    def __init__(self, in_channels, mid_channels, BatchNorm=nn.BatchNorm2d):
        super(context_fusion, self).__init__()
        self.f_x = nn.Sequential(
                                nn.Conv3d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                nn.BatchNorm3d(mid_channels)
                                )
        self.f_y = nn.Sequential(
                                nn.Conv3d(in_channels, mid_channels, 
                                          kernel_size=1, bias=False),
                                nn.BatchNorm3d(mid_channels)
                                )
        
    def forward(self, x, y):
        input_size = x.size()

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3], input_size[4]],
                            mode='trilinear', align_corners=False)
        x_k = self.f_x(x)
        
        sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
        y = F.interpolate(y, size=[input_size[2], input_size[3], input_size[4]],
                            mode='trilinear', align_corners=False)
        x = (1-sim_map)*x + sim_map*y
        
        return x
    
class branch_fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(branch_fusion, self).__init__()
        self.conv_x = nn.Sequential(
                                nn.Conv3d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                nn.BatchNorm3d(out_channels)
                                )
        self.conv_y = nn.Sequential(
                                nn.Conv3d(in_channels, out_channels, 
                                          kernel_size=1, bias=False),
                                nn.BatchNorm3d(out_channels)
                                )
        
    def forward(self, x, y, z):
        z = torch.sigmoid(z)
        out_x = self.conv_x((1-z)*y + x)
        out_y = self.conv_y(y + z*x)
        
        return out_x + out_y
    
    
class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1))

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv +cv
        return cv
    
    
class MMBA(nn.Module):

    def __init__(self, 
        in_channels=8,
        n_channels=8,
        n_classes=1,
        exp_r=4,                            # Expansion ratio as in Swin Transformers
        kernel_size=3,                      # Ofcourse can test kernel_size
        do_res=True,                       # Can be used to individually test residual connection
        do_res_up_down=True,             # Additional 'res' connection on up and down convs
        block_counts: list = [1,1,1,1,1,1,1,1,1], # Can be used to test staging ratio:
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        CGE=True,
    ):

        super().__init__()

        self.CGE = CGE

        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]
            
        self.relu = nn.ReLU(True)
        
        self.encoder_block_0 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=kernel_size,
                do_res=do_res,
                ) 
            for i in range(block_counts[0])]
        ) 

        self.downblock_0 = nn.Sequential(
            BasicConv(n_channels, n_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(n_channels * 2, n_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))
    
        self.encoder_block_1 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[1])]
        )

        self.downblock_1 = nn.Sequential(
            BasicConv(n_channels * 2, n_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(n_channels * 4, n_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.encoder_block_2 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[2])]
        )

        self.downblock_2 = nn.Sequential(
            BasicConv(n_channels * 4, n_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(n_channels * 6, n_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))
        
        self.encoder_block_3 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*6,
                out_channels=n_channels*6,
                exp_r=exp_r[3],
                kernel_size=kernel_size,
                do_res=do_res,
                )            
            for i in range(block_counts[3])]
        )
        
        # self.downblock_3 = nn.Sequential(
        #     BasicConv(n_channels * 6, n_channels * 8, is_3d=True, bn=True, relu=True, kernel_size=3,
        #               padding=1, stride=2, dilation=1),
        #     BasicConv(n_channels * 8, n_channels * 8, is_3d=True, bn=True, relu=True, kernel_size=3,
        #               padding=1, stride=1, dilation=1))

        self.bottleneck = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*6,
                out_channels=n_channels*6,
                exp_r=exp_r[4],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[4])]
        )

        # self.upsample_3 = nn.Sequential(
        #     BasicConv(n_channels * 8, n_channels * 6, deconv=True, is_3d=True, bn=True,
        #                           relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)),
        #     BasicConv(n_channels * 6, n_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
        #               padding=1, stride=1, dilation=1))

        self.decoder_block_3 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*6,
                out_channels=n_channels*6,
                exp_r=exp_r[5],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[5])]
        )

        self.upsample_2 = nn.Sequential(
            BasicConv(n_channels * 6, n_channels * 4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)),
            BasicConv(n_channels * 4, n_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.decoder_block_2 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[6])]
        )

        self.upsample_1 = nn.Sequential(
            BasicConv(n_channels * 4, n_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)),
            BasicConv(n_channels * 2, n_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.decoder_block_1 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[7])]
        )

        self.upsample_0 = nn.Sequential(
            BasicConv(n_channels * 2, n_channels * 1, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2)),
            BasicConv(n_channels * 1, n_channels * 1, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.decoder_block_0 = nn.Sequential(*[
            mambaBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=kernel_size,
                do_res=do_res,
                )
            for i in range(block_counts[8])]
        )

        self.out_0 = nn.Conv3d(n_channels, n_classes, kernel_size=1)

        self.block_counts = block_counts
        
        
        # spatial branch
        self.context_layer1 = BasicBlock3d(inplanes=n_channels*2, planes=n_channels*2)
        self.spatial_convbn1 = nn.Sequential(
                                          nn.Conv3d(n_channels*4, n_channels*2, kernel_size=1, bias=False),
                                          nn.BatchNorm3d(n_channels*2, momentum=0.1),
                                          )
        self.context_fusion1 = context_fusion(n_channels*2, n_channels*2)
        self.context_layer2 = BasicBlock3d(inplanes=n_channels*2, planes=n_channels*2)
        self.spatial_convbn2 = nn.Sequential(
                                          nn.Conv3d(n_channels*6, n_channels*2, kernel_size=1, bias=False),
                                          nn.BatchNorm3d(n_channels*2, momentum=0.1),
                                          )
        self.context_fusion2 = context_fusion(n_channels*2, n_channels*2)
        self.context_layer3 = BasicBlock3d(inplanes=n_channels*2, planes=n_channels*2)

        # edge branch
        self.edge_layer1 = BasicBlock3d(inplanes=n_channels*2, planes=n_channels*2)
        self.edge_convbn1 = nn.Sequential(
                                    nn.Conv3d(n_channels*4, n_channels*2, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm3d(n_channels*2, momentum=0.1),
                                    )
        self.edge_layer2 = BasicBlock3d(inplanes=n_channels*2, planes=n_channels*2)
        self.edge_convbn2 = nn.Sequential(
                                    nn.Conv3d(n_channels*6, n_channels*2, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm3d(n_channels*2, momentum=0.1),
                                    )
        self.edge_layer3 = BasicBlock3d(inplanes=n_channels*2, planes=n_channels*2)
        
        self.branch_fusion = branch_fusion(n_channels*2, n_channels*2)
        
        # attn

        if self.CGE:
            self.feature_att_4 = channelAtt(in_channels * 1, 96)
            self.feature_att_8 = channelAtt(in_channels * 2, 64)
            self.feature_att_16 = channelAtt(in_channels * 4, 192)
            self.feature_att_32 = channelAtt(in_channels * 6, 160)
            self.feature_att_up_32 = channelAtt(in_channels * 6, 160)
            self.feature_att_up_16 = channelAtt(in_channels * 4, 192)
            self.feature_att_up_8 = channelAtt(in_channels * 2, 64)
            self.feature_att_up_4 = channelAtt(in_channels * 1, 96)

    def forward(self, x, imgs, edge_fea):
        x_res_0 = self.encoder_block_0(x)
        x_res_0 = self.feature_att_4(x_res_0, imgs[0])

        x = self.downblock_0(x_res_0)
        x_res_1 = self.encoder_block_1(x)
        x_res_1 = self.feature_att_8(x_res_1, imgs[1])
        
        x8 = x_res_1
        x8_spatial = self.context_layer1(x8)
        x8_size=x8.size()
        edge_fea1 = F.interpolate(edge_fea, size=[x8_size[3], x8_size[4]], mode='bilinear', align_corners=True)
        edge_fea1 = edge_fea1.unsqueeze(1)
        edge = torch.sigmoid(edge_fea1) * x8
        x8_edge = edge + x8
        x8_edge = self.edge_layer1(x8_edge)
        
        x = self.downblock_1(x_res_1)
        x_res_2 = self.encoder_block_2(x)
        x_res_2 = self.feature_att_16(x_res_2, imgs[2])
        
        x16 = x_res_2
        # spatial
        temp = self.spatial_convbn1(x16)
        x8_spatial = self.context_fusion1(x8_spatial, temp)
        x8_spatial = self.context_layer2(self.relu(x8_spatial))
        # edge
        x8_edge_size = x8_edge.size()
        x8_edge = x8_edge + F.interpolate(
                        self.edge_convbn1(x16),
                        size=[x8_edge_size[2], x8_edge_size[3], x8_edge_size[4]],
                        mode='trilinear', align_corners=True)
        x8_edge = self.edge_layer2(self.relu(x8_edge))

        x = self.downblock_2(x_res_2)
        x_res_3 = self.encoder_block_3(x)
        x_res_3 = self.feature_att_32(x_res_3, imgs[3])
        
        x32 = x_res_3
        # spatial
        temp = self.spatial_convbn2(x32)
        x8_spatial = self.context_fusion2(x8_spatial, temp)
        x8_spatial = self.context_layer3(self.relu(x8_spatial))
        # edge
        x8_edge_size = x8_edge.size()
        x8_edge = x8_edge + F.interpolate(
                        self.edge_convbn2(x32),
                        size=[x8_edge_size[2], x8_edge_size[3], x8_edge_size[4]],
                        mode='trilinear', align_corners=True)
        x8_edge = self.edge_layer3(self.relu(x8_edge))

        x = self.bottleneck(x)
        
        x_up_3 = x
        dec_x = x_res_3 + x_up_3
        x = self.decoder_block_3(dec_x)

        x_up_2 = self.upsample_2(x)
        dec_x = x_res_2 + x_up_2
        x = self.decoder_block_2(dec_x)
        x = self.feature_att_up_16(x, imgs[2])

        x_up_1 = self.upsample_1(x)
        dec_x = x_res_1 + x_up_1
        x = self.decoder_block_1(dec_x)
        x = self.feature_att_up_8(x, imgs[1])
        x = self.branch_fusion(x8_spatial, x, x8_edge)
        

        x_up_0 = self.upsample_0(x)
        dec_x = x_res_0 + x_up_0
        x = self.decoder_block_0(dec_x)
        x = self.feature_att_up_4(x, imgs[0])

        x = self.out_0(x)

        return x, x8_spatial, x8_edge


