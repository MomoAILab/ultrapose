"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import torch
from torch import nn
from .trans_unet.vit_seg_modeling import VisionTransformer
from .trans_unet.vit_seg_configs import get_r50_b16_config
from .base_net import BaseNet
from utils.util import transfor_weights

class TransformerNet(BaseNet):
    def __init__(self, lr=1e-4, isTrain=True, device='cpu'):
        super(TransformerNet, self).__init__(lr, isTrain, device)

        config_vit = get_r50_b16_config()
        config_vit.n_skip = 3
        config_vit.patches.grid = (16, 16)
        self.net_g = VisionTransformer(config_vit, img_size=512, in_channels=6, num_classes=15, num_iuv=25).to(self.device)

        if self.isTrain:
            self.modlist = nn.ModuleList([self.net_g])
            self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def forward(self, input_dict):
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].to(self.device)
        s_img, i_img, u_img, v_img = self.net_g(input_dict['input_conditon'])

        _, s_class = torch.max(s_img, 1)

        ret_dict = dict(
            s_img=s_img,
            i_img=i_img,
            u_img=u_img,
            v_img=v_img,
            s_class=s_class,
        )

        if self.isTrain:
            loss_ce = self.criterion_ce(s_img, input_dict['body_s_class'][:, 0].long())

            armMaskLoss, legMaskLoss, trunkMaskLoss = self.calcuMaskLoss(s_class.unsqueeze(1),
                                                                         input_dict['body_s_class'])
            loss_contour = armMaskLoss + legMaskLoss + trunkMaskLoss

            dp_xy_iuv = input_dict['dp_xy_iuv'][:, 0, :, :]
            batch_idx, pt_idx = torch.nonzero(dp_xy_iuv[:, :, 0] > 0.1).unbind(dim=1)
            x = dp_xy_iuv[batch_idx, pt_idx, 1]
            y = dp_xy_iuv[batch_idx, pt_idx, 2]
            slice_index_uv = dp_xy_iuv[batch_idx, pt_idx, 3].long()

            apearenceList = torch.unique(slice_index_uv)
            lossWeights = slice_index_uv.clone().to(torch.float32)
            for itm in apearenceList:
                curIndex = torch.where(slice_index_uv == itm)
                lossWeights[curIndex] = 1. / len(apearenceList) / len(curIndex[0])

            _, _, img_h, img_w = s_img.shape

            # core: layer relationship
            x_lo = x.floor().long().clamp(min=0, max=img_w - 1)
            x_hi = (x_lo + 1).clamp(max=img_w - 1)
            x_w = x - x_lo.float()

            y_lo = y.floor().long().clamp(min=0, max=img_h - 1)
            y_hi = (y_lo + 1).clamp(max=img_h - 1)
            y_w = y - y_lo.float()

            w_ylo_xlo = (1.0 - x_w) * (1.0 - y_w)
            w_ylo_xhi = x_w * (1.0 - y_w)
            w_yhi_xlo = (1.0 - x_w) * y_w
            w_yhi_xhi = x_w * y_w

            u_gt = dp_xy_iuv[batch_idx, pt_idx, 4]
            u_est = self.extract_at_points_packed(u_img, batch_idx, slice_index_uv, y_lo, y_hi,
                                                  x_lo, x_hi, w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi)

            v_gt = dp_xy_iuv[batch_idx, pt_idx, 5]
            v_est = self.extract_at_points_packed(v_img, batch_idx, slice_index_uv, y_lo, y_hi,
                                                  x_lo, x_hi, w_ylo_xlo, w_ylo_xhi, w_yhi_xlo, w_yhi_xhi)

            i_est = self.extract_at_points_packed(i_img, batch_idx, slice(None), y_lo, y_hi, x_lo, x_hi,
                                                  w_ylo_xlo[:, None], w_ylo_xhi[:, None], w_yhi_xlo[:, None],
                                                  w_yhi_xhi[:, None])

            loss_u = self.criterion_l1(u_est, u_gt)
            loss_u = torch.sum(loss_u * lossWeights)
            loss_v = self.criterion_l1(v_est, v_gt)
            loss_v = torch.sum(loss_v * lossWeights)
            loss_i = self.criterion_ce(i_est, slice_index_uv)

            ret_dict['loss_dict'] = dict(
                ce=loss_ce*10,
                i=loss_i*5,
                u=loss_u*400,
                v=loss_v*400,
                contour=loss_contour * 0.005,
            )
        return ret_dict

    def get_weights(self):
        ret_dict = dict(
            net_g=self.net_g.state_dict(),
        )
        return ret_dict

    def set_weights(self, wdict):
        transfor_weights(self.net_g, wdict['net_g'])

def create_transformer_net(args, lr, isTrain, device):
    return TransformerNet(lr, isTrain, device)

