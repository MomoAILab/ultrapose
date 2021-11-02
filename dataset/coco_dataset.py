"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import random
import cv2
import numpy as np
import os.path as osp
import math
import json
import pycocotools.mask as mask_util
from .base_dataset import BaseDataset

cur_dir = osp.dirname(__file__)

distinct_color_map = [[0, 0, 0], [255, 200, 200], [255, 0, 200], [255, 0, 100], [200, 0, 255], [255, 0, 0],
                           [0, 0, 255],
                           [0, 255, 0], [0, 100, 255], [255, 155, 100], [255, 200, 255], [0, 200, 255], [100, 0, 0],
                           [100, 100, 0], [0, 100, 100], [50, 100, 255]]

class COCODataLoader(BaseDataset):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

        self.output_img_size = (512, 512)
        self.isTrain = is_train

        self.all_data, self.image_path_dict = self.load_coco_datum(self.dataroot, self.isTrain)

        if len(self.all_data) == 0 :
            print(" NO DATA FILES!")
            exit(-1)

        if self.isTrain:
            random.seed(args.seed)
            random.shuffle(self.all_data)

        print('(rank{}), All Data Num: {}'.format(args.rank, len(self.all_data)))

        self.seg2index_table_cd = [[], [1, 2], [3], [4], [5], [6], [7, 9], [8, 10], [11, 13], [12, 14],
                                [15, 17], [16, 18], [19, 21], [20, 22], [23, 24]]


    def __getitem__(self, index):
        annotation = self.all_data[index]
        image_path = self.image_path_dict[annotation['image_id']]
        input_data = self.get_data(annotation, image_path)

        origin_img = input_data['image']
        kpt = input_data['kpnts17']
        dp_mask = input_data['dp_mask']
        xyiuv = input_data['xyiuv']

        img_w, img_h = self.output_img_size

        downLength = (kpt[16][1] - kpt[12][1] + kpt[15][1] - kpt[11][1]) / 2
        kneeLength = (kpt[14][1] - kpt[12][1] + kpt[13][1] - kpt[11][1]) / 2
        upLength = (kpt[12][1] - kpt[6][1] + kpt[11][1] - kpt[5][1]) / 2

        kneeBottom = -1
        hipBottom = -1
        if downLength > upLength * 1.6 and upLength > 60 and random.random() < 0.1:
            # cover knee
            kneeBottom = min(kpt[14][1], kpt[13][1])
            kneeBottom = int(kneeBottom + random.random() * 30 - 15)
            origin_img[kneeBottom:, :, :] = 0

            if kpt[13][1] > kneeBottom: kpt[13][2] = 0.
            if kpt[14][1] > kneeBottom: kpt[14][2] = 0.
            kpt[15][2] = 0.
            kpt[16][2] = 0.

        if kneeLength > upLength * 0.85 and upLength > 60 and random.random() < 0.08:
            # cover hip
            hipBottom = min(kpt[11][1], kpt[12][1])
            hipBottom = int(hipBottom + random.random() * 30)
            origin_img[hipBottom:, :, :] = 0

            if kpt[13][1] > hipBottom: kpt[13][2] = 0.
            if kpt[14][1] > hipBottom: kpt[14][2] = 0.
            kpt[15][2] = 0.
            kpt[16][2] = 0.

        kpnts17 = kpt.copy()
        kpnts17_img = self.draw_connect_keypoints([kpnts17], 512, 512).astype(np.float32) / 255

        if self.isTrain and random.randint(0, 3):
            origin_img = self.seq(image=origin_img)
        main_image = origin_img.astype(np.float32) / 255
        if dp_mask is None:
            body_s_class = np.zeros(self.output_img_size, dtype=np.float32)[:,:,np.newaxis]
        else:
            body_s_class = dp_mask
            if hipBottom > 0:
                body_s_class[hipBottom:, :] = 0
                sClassSegImg = self.get_render_img(body_s_class[:, :])
                body_s_class = self.seg2s(sClassSegImg)[:, :, np.newaxis]
            elif kneeBottom > 0:
                body_s_class[kneeBottom:, :] = 0
                sClassSegImg = self.get_render_img(body_s_class[:, :])
                body_s_class = self.seg2s(sClassSegImg)[:, :, np.newaxis]
            else:
                sClassSegImg = self.get_render_img(body_s_class[:, :])
                body_s_class = self.seg2s(sClassSegImg)[:, :, np.newaxis]

        dp_xy_iuv = np.zeros([3000, 6], dtype=np.float32)

        val_cnt = 0
        for idx, (pt_x, pt_y, I, U, V) in enumerate(xyiuv):
            pt_x, pt_y = int(pt_x), int(pt_y)
            if pt_x < 0 or pt_x >= img_w - 1: continue
            if pt_y < 0 or pt_y >= img_h - 1: continue
            if hipBottom > 0 and pt_y >= hipBottom - 1: continue
            if kneeBottom > 0 and pt_y >= kneeBottom - 1: continue
            if idx >= 3000: continue
            dp_xy_iuv[val_cnt] = [1.0, pt_x, pt_y, I, U, V]
            val_cnt += 1

        dp_xy_iuv = np.expand_dims(dp_xy_iuv, axis=2)

        input_conditon = np.concatenate([main_image, kpnts17_img], axis=2)
        ret_dic = {'input_conditon': input_conditon,
                   'body_s_class': body_s_class,
                   'mask_weights': np.zeros((1, 1, 1), dtype=np.float32) if dp_mask is None else np.ones((1, 1, 1), dtype=np.float32),
                   'dp_xy_iuv': dp_xy_iuv,
                   }
        ret_dic = {k: v.transpose([2, 0, 1]) for k, v in ret_dic.items()}
        return ret_dic

    def __len__(self):
        return len(self.all_data)

    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def get_data(self, item, image_path):

        image = cv2.imread(image_path)
        if 'keypoints' in item:
            kpnts17 = np.array(item['keypoints']).reshape(-1, 3)
        else:
            kpnts17 = np.zeros((17, 3), dtype=np.float32)

        (bx, by, bw, bh) = [int(x) for x in item['bbox']]

        rand_ratio, rand_offsetx, rand_offsety, rand_rot = 0.9, 0, 0, 0
        if self.isTrain:
            rand_ratio = random.uniform(0.7, 1.3)
            rand_offsetx = random.uniform(-5, 5)
            rand_offsety = random.uniform(-5, 5)
            rand_rot = math.degrees((random.uniform(-25, 25) / 180) * math.pi)

        bbox = np.array([bx, by, bx+bw-1, by+bh-1], dtype=np.float32)
        cx, cy = bx + bw * 0.5, by + bh * 0.5
        ratio = min(self.output_img_size[0] / bw, self.output_img_size[1] / bh) * rand_ratio
        M_image = cv2.getRotationMatrix2D((cx, cy), rand_rot, ratio)
        M_image[:, 2] += np.array([self.output_img_size[0]*0.5 - cx + rand_offsetx,
                                   self.output_img_size[1]*0.5 - cy + rand_offsety])

        M_label = cv2.getRotationMatrix2D((bw*0.5, bh*0.5), rand_rot, ratio)
        M_label[:, 2] += np.array([self.output_img_size[0]*0.5 - bw*0.5 + rand_offsetx,
                                   self.output_img_size[1]*0.5 - bh*0.5 + rand_offsety])

        image = cv2.warpAffine(image, M_image, self.output_img_size)
        bbox = self.transform_points(bbox.reshape(-1, 2), M_image).reshape(-1).astype(np.int)

        if 'dp_masks' in item:
            dp_mask = COCODataLoader.GetDensePoseMask(item['dp_masks'])
            dp_mask = cv2.warpAffine(cv2.resize(dp_mask, (bw, bh), interpolation=cv2.INTER_NEAREST), M_label, self.output_img_size, flags=cv2.INTER_NEAREST)
        else:
            dp_mask = None

        xyiuv = []
        try:
            for idx in range(len(item['dp_x'])):
                x, y = item['dp_x'][idx] / 256 * bw, item['dp_y'][idx] / 256 * bh
                i, u, v = item['dp_I'][idx], item['dp_U'][idx], item['dp_V'][idx]
                xyiuv.append([x, y, i, u, v])

            xyiuv = np.array(xyiuv, dtype=np.float32)
            xyiuv[:, :2] = self.transform_points(xyiuv[:, :2], M_label)
        except:
            xyiuv=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            xyiuv = np.array(xyiuv, dtype=np.float32)
            xyiuv[:,:2] = self.transform_points(xyiuv[:,:2], M_label)

        kpnts17[:, :] -= np.array([[bx, by, 0]], dtype=np.int)
        kpnts17[:,:2] = self.transform_points(kpnts17[:,:2], M_label)

        if self.isTrain:
            kpnts17 = self.rand_kpnts(image.shape[1], image.shape[0], kpnts17)

        ret_dict = dict(
            image=image,
            dp_mask=dp_mask,
            xyiuv=xyiuv,
            kpnts17=kpnts17,
            bbox=bbox,
        )

        return ret_dict

    @staticmethod
    def GetDensePoseMask(Polys, canvasSize=256):
        MaskGen = np.zeros([canvasSize, canvasSize])
        for i in range(1, 15):
            if (Polys[i - 1]):
                current_mask = mask_util.decode(Polys[i - 1])
                MaskGen[current_mask > 0] = i
        return MaskGen

    @staticmethod
    def load_coco_datum(data_root, is_train):
        anno_path = osp.join(data_root,
                             'annotations/densepose_%s2014.json' % ('train' if is_train else 'valminusminival'))
        anno_params = json.load(open(anno_path))
        image_path_dict = dict()
        for item in anno_params['images']:
            image_path = osp.join(data_root, '%s2014' % ('train' if is_train else 'val'), item['file_name'])
            image_path_dict[item['id']] = image_path

        selected_data_list = []
        for item in anno_params['annotations']:
            if 'keypoints' in item and 'dp_I' in item and np.sum(np.array(item['keypoints']).reshape(-1, 3)[:,2]>0.2) > 10:
                selected_data_list.append(item)
        return selected_data_list, image_path_dict

    @staticmethod
    def get_render_img(seg_class):
        color_list = distinct_color_map
        if not isinstance(seg_class, np.ndarray): seg_class = seg_class.detach().cpu().numpy().astype(np.int)
        seg_img = np.zeros((512, 512, 3), dtype=np.uint8)

        for idx in range(15):
            seg_img[seg_class == idx] = color_list[idx]

        return seg_img

    def seg2s(self, segImg):
        color_list = distinct_color_map

        body_s_class = np.zeros((segImg.shape[0], segImg.shape[1]), dtype=np.float32)

        for idx in range(15):
            curIndex = (segImg == [color_list[idx][0], color_list[idx][1], color_list[idx][2]]).all(axis=2).astype(
                np.uint8)
            body_s_class[curIndex != 0] = idx
        return body_s_class

    def gen_iuv(self, image, xyiuv, ratio=1.):
        ret_img = (image.copy() * ratio).astype(np.uint8)
        for _, x, y, i, u, v in xyiuv:
            cv2.circle(ret_img, (int(x), int(y)), 2, (int(i), int(u*255), int(v*255)), -1)
        return ret_img

    def get_dp_img(self, s_img, i_img, u_img, v_img):
        w = 512
        h = 512
        s_img_np = s_img.astype(np.int32)
        i_img_np = i_img
        u_img_np = u_img
        v_img_np = v_img

        dp_img = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                max_idx = np.argmax(i_img_np[:,i,j])

                if max_idx > 0 and s_img_np[i,j] > 0:
                    dp_img[i, j, 0] = max_idx
                    dp_img[i, j, 1] = min(u_img_np[max_idx, i, j] * 255, 255)
                    dp_img[i, j, 2] = min(v_img_np[max_idx, i, j] * 255, 255)
        dp_img = dp_img.astype(np.uint8)

        seg_color = np.zeros((h, w, 3), dtype=np.uint8)
        for part_id in range(1, 15):
            index = (s_img_np == part_id)
            seg_color[index] = distinct_color_map[part_id]

        return dp_img, seg_color

    def cat_imgs(self, img, ret, ratio=1.):
        disp = (img.copy() * ratio).astype(np.uint8)
        mask = ret > 0
        disp[mask] = ret[mask]
        return disp

    def show_training_results(self, sample, data_dict=None, index=0):
        image = (sample['input_conditon'][index][:3].detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        disp = image.copy()
        image = self.put_text(self.resize_256(image), (5, 15), "image")

        bg_ratio = 0.6

        skl = sample['input_conditon'][index][3:].detach().cpu().numpy().transpose(1, 2, 0)*255
        skl = self.cat_imgs(disp, skl, bg_ratio)
        skl = self.put_text(self.resize_256(skl), (5, 15), "skl")

        seg = self.get_render_img(sample['body_s_class'][index][0].detach().cpu().numpy().astype(np.int))
        seg = self.cat_imgs(disp, seg, bg_ratio)
        seg = self.put_text(self.resize_256(seg), (5, 15), "gt_seg", color=(0, 128, 0))

        iuv = self.gen_iuv(disp, sample['dp_xy_iuv'][index][0].detach().cpu().numpy(), bg_ratio)
        iuv = self.put_text(self.resize_256(iuv), (5, 15), "gt_iuv", color=(0, 128, 0))

        show = np.concatenate((image, skl, seg, iuv), axis=1)
        if not data_dict is None:
            i_img = data_dict['i_img'][index].detach().cpu().numpy()
            u_img = data_dict['u_img'][index].detach().cpu().numpy()
            v_img = data_dict['v_img'][index].detach().cpu().numpy()
            s_class = data_dict['s_class'][index].detach().cpu().numpy()

            pred_iuv, pred_seg = self.get_dp_img(s_class, i_img, u_img, v_img)
            pred_iuv = self.cat_imgs(disp, pred_iuv, bg_ratio)
            pred_seg = self.cat_imgs(disp, pred_seg, bg_ratio)
            pred_seg = self.put_text(self.resize_256(pred_seg), (5, 15), "pred_seg", color=(0, 0, 128), )
            pred_iuv = self.put_text(self.resize_256(pred_iuv), (5, 15), "pred_iuv", color=(0, 0, 128))
            show = np.concatenate((show, pred_seg, pred_iuv), axis=1)
        return show