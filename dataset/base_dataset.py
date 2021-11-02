"""
created by songkey@pku.edu.cn
date @ 2021-10-15
"""
import torch.utils.data as data
from abc import ABC, abstractmethod
import cv2
import numpy as np
import os.path as osp
import random
import imgaug.augmenters as iaa

cur_dir = osp.dirname(__file__)

class BaseDataset(data.Dataset, ABC):
    def __init__(self, args, is_train=False):
        self.args = args
        self.isTrain = is_train

        if is_train:
            self.dataroot = args.dataroot
        else:
            self.dataroot = args.eval_dataroot

        if self.isTrain:
            sometimes = lambda aug: iaa.Sometimes(0.5, aug)
            self.seq = iaa.Sequential(
                [
                    iaa.SomeOf((0, 5),
                               [
                                   iaa.OneOf([
                                       iaa.GaussianBlur((0, 3.0)),
                                       iaa.AverageBlur(k=(2, 3)),
                                       iaa.MedianBlur(k=(3, 3)),
                                   ]),
                                   iaa.Sharpen(alpha=(0, 0.5), lightness=(0.25, 0.5)),
                                   iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.)),
                                   sometimes(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0, 0.7)),
                                       iaa.DirectedEdgeDetect(
                                           alpha=(0, 0.7), direction=(0.0, 1.0)
                                       ),
                                   ])),

                                   iaa.AdditiveGaussianNoise(
                                       loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                                   ),
                                   iaa.OneOf([
                                       iaa.Dropout((0.01, 0.1), per_channel=0.5),
                                       iaa.CoarseDropout(
                                           (0.03, 0.15), size_percent=(0.02, 0.05),
                                           per_channel=0.2
                                       ),
                                   ]),
                                   iaa.Invert(0.05, per_channel=True),
                                   iaa.Add((-10, 10), per_channel=0.5),
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                                   iaa.LinearContrast((0.5, 1.0), per_channel=0.5),

                                   iaa.Grayscale(alpha=(0.0, 1.0)),

                               ],
                               random_order=True
                               )
                ],
                random_order=True
            )

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

    def get_net_datapool(self):
        pass

    @staticmethod
    def put_text(img, org, text, color=(128, 128, 0)):
        disp = img.copy()
        cv2.putText(disp, text, org, 3, 0.4, (255, 255, 255), 3)
        cv2.putText(disp, text, org, 3, 0.4, color, 1)
        return disp

    @staticmethod
    def rand_kpnts(w, h, kpnts):
        rand_kpnts = kpnts.copy()
        selj = kpnts[kpnts[:,2] > 0.1, :]
        areahw = np.max(selj[:, :2], axis=0) - np.min(selj[:, :2], axis=0)
        gt_area = areahw[0] * areahw[1]
        body_ratio = np.sqrt(gt_area) / min(h, w) * 0.5
        offset = body_ratio * min(h, w) * 0.09 / np.sqrt(2) * 0.5
        for idx in range(rand_kpnts.shape[0]):
            offsetx = random.uniform(-offset, offset) * (3 - 2)
            offsety = random.uniform(-offset, offset) * (3 - 2)
            rand_kpnts[idx, 0] += offsetx
            rand_kpnts[idx, 1] += offsety
        return rand_kpnts

    @staticmethod
    def resize_256(img):
        img = cv2.resize(img, (256, 256))
        return img

    @staticmethod
    def draw_connect_keypoints(keypoints, w, h, draw_line=True):
        COCO_PERSON_KEYPOINT_NAMES = (
            "nose",
            "left_eye", "right_eye",
            "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        )
        COCO_PERSON_KEYPOINT_COLORS = (
            (102, 204, 255),
            (51, 153, 255),
            (102, 0, 204),
            (51, 102, 255),
            (153, 255, 204),
            (128, 229, 255),
            (153, 255, 153),
            (102, 255, 224),
            (255, 102, 0),
            (255, 255, 77),
            (153, 255, 204),
            (191, 255, 128),
            (255, 195, 77),
            (77, 204, 22),
            (22, 139, 77),
            (0, 56, 138),
            (138, 76, 23),
        )
        KEYPOINT_CONNECTION_RULES = [
            # face
            ("left_ear", "left_eye", (102, 204, 255)),
            ("right_ear", "right_eye", (51, 153, 255)),
            ("left_eye", "nose", (102, 0, 204)),
            ("nose", "right_eye", (51, 102, 255)),
            # upper-body
            ("left_shoulder", "right_shoulder", (255, 128, 0)),
            ("left_shoulder", "left_elbow", (153, 255, 204)),
            ("right_shoulder", "right_elbow", (128, 229, 255)),
            ("left_elbow", "left_wrist", (153, 255, 153)),
            ("right_elbow", "right_wrist", (102, 255, 224)),
            # lower-body
            ("left_hip", "right_hip", (255, 102, 0)),
            ("left_hip", "left_knee", (255, 255, 77)),
            ("right_hip", "right_knee", (153, 255, 204)),
            ("left_knee", "left_ankle", (191, 255, 128)),
            ("right_knee", "right_ankle", (255, 195, 77)),
        ]
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for ix, keypoint in enumerate(keypoints):
            visible = {}
            for idx, pt in enumerate(keypoint):
                x, y, prob = pt
                keypoint_name = COCO_PERSON_KEYPOINT_NAMES[idx]
                if x < 0 or x >= w: continue
                if y < 0 or y >= h: continue
                if prob < 0.3: continue
                visible[keypoint_name] = (int(x), int(y), prob)

            for ix, (k, v) in enumerate(visible.items()):
                if not draw_line and (
                        k == 'nose' or k == 'left_eye' or k == 'right_eye' or k == 'left_ear' or k == 'right_ear'):
                    continue
                cv2.circle(image, (v[0], v[1]), 5, color=COCO_PERSON_KEYPOINT_COLORS[ix], thickness=cv2.FILLED)
                # cv2.putText(image, '{}%'.format(int(v[2] * 100)), (int(v[0] + 5), int(v[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                #             (0, 255, 0), 1)

            if not draw_line:
                continue

            for kp0, kp1, color in KEYPOINT_CONNECTION_RULES:
                if kp0 in visible and kp1 in visible:
                    x0, y0, _ = visible[kp0]
                    x1, y1, _ = visible[kp1]
                    color = (color[2], color[1], color[0])
                    cv2.line(image, (x0, y0), (x1, y1), color=color, thickness=5, lineType=cv2.LINE_AA)
            try:
                ls_x, ls_y, _ = visible["left_shoulder"]
                rs_x, rs_y, _ = visible["right_shoulder"]
                mid_shoulder_x, mid_shoulder_y = int((ls_x + rs_x) / 2), int((ls_y + rs_y) / 2)
            except KeyError:
                pass
            else:
                # draw line from nose to mid-shoulder
                nose_x, nose_y, _ = visible.get("nose", (None, None, None))
                if nose_x is not None:
                    cv2.line(image, (nose_x, nose_y), (mid_shoulder_x, mid_shoulder_y), color=(0, 0, 255), thickness=5,
                             lineType=cv2.LINE_AA)

                try:
                    # draw line from mid-shoulder to mid-hip
                    lh_x, lh_y, _ = visible["left_hip"]
                    rh_x, rh_y, _ = visible["right_hip"]
                except KeyError:
                    pass
                else:
                    mid_hip_x, mid_hip_y = int((lh_x + rh_x) / 2), int((lh_y + rh_y) / 2)
                    cv2.line(image, (mid_hip_x, mid_hip_y), (mid_shoulder_x, mid_shoulder_y), color=(0, 0, 255),
                             thickness=5,
                             lineType=cv2.LINE_AA)
        return image

    @staticmethod
    def transform_points(points, mat, invert=False):
        if invert:
            mat = cv2.invertAffineTransform(mat)
        points = np.expand_dims(points, axis=1)
        points = cv2.transform(points, mat, points.shape)
        points = np.squeeze(points)
        return points


