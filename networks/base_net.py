"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from loss.dp_loss import SoftIOULoss

class BaseNet(nn.Module, ABC):
    def __init__(self, lr=1e-4, isTrain=True, device='cpu'):
        super(BaseNet, self).__init__()
        self.device = device
        self.isTrain = isTrain
        self.lr = lr

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
            self.criterion_l1 = torch.nn.SmoothL1Loss(reduction="none")
            self.criterion_l2 = torch.nn.MSELoss()
            self.criterion_iou = SoftIOULoss(weight=2)

    @abstractmethod
    def set_weights(self, wdict):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    def load_checkpoint(self, visualizer, args):
        weights = torch.load(args.finetune_model_path, map_location="cpu")
        self.set_weights(weights)

        if not visualizer is None:
            visualizer.load_data(weights)
        return weights['epoch'], \
               weights['learning_rate'], \
               weights['it_start'] if 'it_start' in weights else 0

    def checkpoint(self, epoch, it_start, visualizer, learning_rate, save_path):
        save_dict = {
            'epoch': epoch + 1,
            'it_start': it_start + 1,
            'visdom_data': None if visualizer is None else visualizer.data,
            'learning_rate': learning_rate,
        }
        save_dict.update(self.get_weights())
        torch.save(save_dict, save_path, _use_new_zipfile_serialization=False)
        print("Checkpoint saved to {}".format(save_path))

    def active_contour_loss(self, y_true, y_pred, weight=10):
        '''
        y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
        delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

        delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
        delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
        lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.

        # region term
        c_in = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)

        region_in = torch.mean(y_pred * (y_true - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean((1 - y_pred) * (y_true - c_out) ** 2)
        region = region_in + region_out

        loss = weight * lenth + region
        return loss

    def calcuMaskLoss(self, s_pred, s_gt):
        batchNum = s_pred.shape[0]
        allWhiteBoard = torch.ones([s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        allBlackBoard = torch.zeros([s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)

        armMask_pred = torch.zeros([batchNum, 1, s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        armMask_gt = torch.zeros([batchNum, 1, s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        legMask_pred = torch.zeros([batchNum, 1, s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        legMask_gt = torch.zeros([batchNum, 1, s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        trunkMask_pred = torch.zeros([batchNum, 1, s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        trunkMask_gt = torch.zeros([batchNum, 1, s_gt.shape[2], s_gt.shape[3]], dtype=torch.float32).to(s_pred.device)
        for bid in range(batchNum):
            armIndex = torch.where(s_pred[bid][0] == 5, allWhiteBoard, allBlackBoard)
            armIndex = torch.where(s_pred[bid][0] == 6, allWhiteBoard, armIndex)
            armIndex = torch.where(s_pred[bid][0] == 7, allWhiteBoard, armIndex)
            armIndex = torch.where(s_pred[bid][0] == 8, allWhiteBoard, armIndex)
            armIndex_gt = torch.where(s_gt[bid][0] == 5, allWhiteBoard, allBlackBoard)
            armIndex_gt = torch.where(s_gt[bid][0] == 6, allWhiteBoard, armIndex_gt)
            armIndex_gt = torch.where(s_gt[bid][0] == 7, allWhiteBoard, armIndex_gt)
            armIndex_gt = torch.where(s_gt[bid][0] == 8, allWhiteBoard, armIndex_gt)
            armMask_pred[bid, 0, :, :] = armIndex
            armMask_gt[bid, 0, :, :] = armIndex_gt

            legIndex = torch.where(s_pred[bid][0] == 9, allWhiteBoard, allBlackBoard)
            legIndex = torch.where(s_pred[bid][0] == 11, allWhiteBoard, legIndex)
            legIndex = torch.where(s_pred[bid][0] == 12, allWhiteBoard, legIndex)
            legIndex = torch.where(s_pred[bid][0] == 14, allWhiteBoard, legIndex)
            legIndex_gt = torch.where(s_gt[bid][0] == 9, allWhiteBoard, allBlackBoard)
            legIndex_gt = torch.where(s_gt[bid][0] == 11, allWhiteBoard, legIndex_gt)
            legIndex_gt = torch.where(s_gt[bid][0] == 12, allWhiteBoard, legIndex_gt)
            legIndex_gt = torch.where(s_gt[bid][0] == 14, allWhiteBoard, legIndex_gt)
            legMask_pred[bid, 0, :, :] = legIndex
            legMask_gt[bid, 0, :, :] = legIndex_gt

            trunkIndex = torch.where(s_pred[bid][0] == 3, allWhiteBoard, allBlackBoard)
            trunkIndex_gt = torch.where(s_gt[bid][0] == 3, allWhiteBoard, allBlackBoard)
            trunkMask_pred[bid, 0, :, :] = trunkIndex
            trunkMask_gt[bid, 0, :, :] = trunkIndex_gt

        return self.criterion_iou(armMask_pred, armMask_gt) * 20, \
               self.criterion_iou(legMask_pred, legMask_gt) * 20, \
               self.criterion_iou(trunkMask_pred, trunkMask_gt) * 20

    @staticmethod
    def extract_at_points_packed(z_est,
                                 batch_idx,
                                 slice_index_uv,
                                 y_lo,
                                 y_hi,
                                 x_lo,
                                 x_hi,
                                 w_ylo_xlo,
                                 w_ylo_xhi,
                                 w_yhi_xlo,
                                 w_yhi_xhi):
        z_est_sampled = (
                z_est[batch_idx, slice_index_uv, y_lo, x_lo] * w_ylo_xlo
                + z_est[batch_idx, slice_index_uv, y_lo, x_hi] * w_ylo_xhi
                + z_est[batch_idx, slice_index_uv, y_hi, x_lo] * w_yhi_xlo
                + z_est[batch_idx, slice_index_uv, y_hi, x_hi] * w_yhi_xhi
        )

        return z_est_sampled
