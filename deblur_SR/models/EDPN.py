#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EDPN.py
@Time    :   2021/05/07 19:54:49
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   the architecture for EDPN
'''
# here put the import lib

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.arch_util as arch_util
try:
    from models.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


class SPA_base(nn.Module):
    def __init__(self, arg):
        super(SPA_base, self).__init__()
        embed_ch = arg.embed_ch
        groups = arg.groups

        self.conv_base = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)
        self.conv_ref = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.offset_conv1 = nn.Conv2d(embed_ch * 2, embed_ch, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)
        self.dcnpack = DCN(embed_ch, embed_ch, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.dcn_conv = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)
        self.softmax = nn.Softmax(-1)

        self.out_conv1 = nn.Conv2d(embed_ch*2, embed_ch, 3, 1, 1, bias=True)
        self.out_conv2 = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x_base, x_ref):
        feat_base = self.conv_base(x_base)
        feat_ref = self.conv_ref(x_ref)

        offset = torch.cat([feat_base, feat_ref], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        feat_Dconv = self.lrelu(self.dcnpack([feat_ref, offset]))
        feat_Dconv = self.lrelu(self.dcn_conv(feat_Dconv))

        feat_motion = feat_ref - feat_base
        atten_motion = self.softmax(feat_motion)

        feat_add = atten_motion * feat_Dconv

        feat_out = torch.cat([feat_base, feat_add], dim=1)
        feat_out = self.lrelu(self.out_conv1(feat_out))
        feat_out = self.lrelu(self.out_conv2(feat_out))
        feat_out = feat_out + x_base

        return feat_out


class SPA_module(nn.Module):
    def __init__(self, arg):
        super(SPA_module, self).__init__()
        self.n_SPA_blocks = arg.n_SPA_blocks
        embed_ch = arg.embed_ch

        SPA_blocks = []
        for _ in range(self.n_SPA_blocks):
            SPA_blocks.append(SPA_base(arg))
        self.SPA_blocks = nn.Sequential(*SPA_blocks)

        self.out_conv1 = nn.Conv2d(embed_ch * 2, embed_ch, 3, 1, 1, bias=True)
        self.out_conv2 = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x_base, x_ref):

        feat_enhance = self.SPA_blocks[0](x_base, x_ref)
        for i in range(1, self.n_SPA_blocks):
            feat_enhance = self.SPA_blocks[i](x_base, feat_enhance)

        feat_out = torch.cat([x_base, feat_enhance], dim=1)
        feat_out = self.lrelu(self.out_conv1(feat_out))
        feat_out = self.lrelu(self.out_conv2(feat_out))
        feat_out = feat_out + x_base

        return feat_out


class PSPA_module(nn.Module):
    def __init__(self, arg):
        super(PSPA_module, self).__init__()
        embed_ch = arg.embed_ch

        self.L3_spa = SPA_module(arg)
        self.L3_conv = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.L2_spa = SPA_module(arg)
        self.L2_conv = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.L1_spa = SPA_module(arg)
        self.L1_conv = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.L2_conv_up1 = nn.Conv2d(embed_ch*2, embed_ch, 3, 1, 1, bias=True)
        self.L2_conv_up2 = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.L1_conv_up1 = nn.Conv2d(embed_ch*2, embed_ch, 3, 1, 1, bias=True)
        self.L1_conv_up2 = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.out_conv1 = nn.Conv2d(embed_ch*2, embed_ch, 3, 1, 1, bias=True)
        self.out_conv2 = nn.Conv2d(embed_ch, embed_ch, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, feats_base, feats_ref):
        L3_feat_base = feats_base[2]
        L3_feat_ref = feats_ref[2]
        L3_feat = self.lrelu(self.L3_conv(self.L3_spa(L3_feat_base, L3_feat_ref)))

        L2_feat_base = feats_base[1]
        L2_feat_ref = feats_ref[1]
        L2_feat = self.lrelu(self.L3_conv(self.L3_spa(L2_feat_base, L2_feat_ref)))

        L1_feat_base = feats_base[0]
        L1_feat_ref = feats_ref[0]
        L1_feat = self.lrelu(self.L3_conv(self.L3_spa(L1_feat_base, L1_feat_ref)))

        L3_fea_up = F.interpolate(L3_feat, scale_factor=2, mode='bilinear', align_corners=False)
        L2_cat = self.lrelu(self.L2_conv_up1(torch.cat([L2_feat, L3_fea_up], dim=1)))
        L2_cat = self.lrelu(self.L2_conv_up2(L2_cat))

        L2_fea_up = F.interpolate(L2_cat, scale_factor=2, mode='bilinear', align_corners=False)
        L1_cat = self.lrelu(self.L1_conv_up1(torch.cat([L1_feat, L2_fea_up], dim=1)))
        L1_cat = self.lrelu(self.L1_conv_up2(L1_cat))

        feat_out = self.lrelu(self.out_conv1(torch.cat([L1_feat_base, L1_cat], dim=1)))
        feat_out = self.lrelu(self.out_conv2(feat_out))

        return feat_out

class TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf, nframes, center):
        super(TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob
        # print(aligned_fea.shape)

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class ResidualGroup_RCAB_MSCAM(nn.Module):
    def __init__(self, n_resblocks):
        super(ResidualGroup_RCAB_MSCAM, self).__init__()
        modules_body = []
        modules_body = [
            arch_util.RCAB_MSCAM() for _ in range(n_resblocks)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)


class TSA3DFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.
    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)
    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSA3DFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # fusion 3D_conv
        self.conv3D = nn.Conv3d(num_frame, num_frame, kernel_size=1, bias=True)
        self.conv3D_fusion = nn.Conv2d(num_frame * num_feat, num_feat, kernel_size=1, bias=True)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).
        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))
        fea_3d = self.lrelu(self.conv3D(aligned_feat.view(b, t, -1, h, w)))
        # print(fea_3d.shape)
        # print(fea_3d.view(B, -1, H, W).shape)
        fea_3d = self.lrelu(self.conv3D_fusion(fea_3d.view(b, -1, h, w)))

        feat = feat + fea_3d
        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat



class EDPN(nn.Module):
    def __init__(self, arg):
        super(EDPN, self).__init__()
        self.nf = arg.embed_ch
        self.nframes = arg.nframes
        self.center = int((self.nframes/2)//2)
        self.groups = arg.groups
        self.front_RBs = arg.front_RBs
        self.back_RBs = arg.back_RBs
        upscale_factor=4

        ResidualBlock_noBN = functools.partial(arch_util.ResidualBlock_noBN, nf=self.nf)

        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(3, self.nf, 3, 1, 1, bias=True)
        self.feature_extractor1 = arch_util.make_layer(ResidualBlock_noBN, self.front_RBs//3)
        self.feature_extractor2 = arch_util.make_layer(ResidualBlock_noBN, self.front_RBs//3)
        self.feature_extractor3 = arch_util.make_layer(ResidualBlock_noBN, self.front_RBs//3)


        self.fea_L2_conv1 = nn.Conv2d(self.nf, self.nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(self.nf, self.nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

        ###############################################################
        self.pspa = PSPA_module(arg)

        ###############################################################
        self.tsa_fusion = TSA3DFusion(num_feat=self.nf, num_frame=self.nframes, center_frame_idx=self.center)

        ###############################################################
        #### reconstruction backbone
        self.reconstructor1 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor2 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor3 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor4 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor5 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor6 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor7 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor8 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor9 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor10 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor11 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)
        self.reconstructor12 = ResidualGroup_RCAB_MSCAM(n_resblocks=10)

        ###############################################################
        #### upsampling
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=True),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=True))
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea1 = self.feature_extractor1(L1_fea)
        L1_fea2 = self.feature_extractor2(L1_fea1)
        L1_fea3 = self.feature_extractor3(L1_fea2)
        L1_fea = L1_fea

        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea_center = L1_fea[:, self.center, :, :, :].contiguous()

        # align and fusion
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pspa(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        ###########################################################################################
        fea = self.tsa_fusion(aligned_fea)

        ###########################################################################################
        out1 = self.reconstructor1(fea)
        out2 = self.reconstructor2(out1)
        out3 = self.reconstructor3(out2)
        out4 = self.reconstructor4(out3)
        out5 = self.reconstructor5(out4)
        out6 = self.reconstructor6(out5)
        out7 = self.reconstructor7(out6)
        out8 = self.reconstructor8(out7)
        out9 = self.reconstructor9(out8)
        out10 = self.reconstructor10(out9)
        out11 = self.reconstructor11(out10)
        out12 = self.reconstructor12(out11)



        ###########################################################################################
        out = self.lrelu(self.HRconv(out12))
        out = self.upscale(out)

        x_center = F.interpolate(x_center, scale_factor=4, mode='bicubic', align_corners=False)
        out += x_center
        return out


