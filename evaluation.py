"""
created by double4tar@gmail.com
date @ 2021-10-15
"""
import os
import cv2
import torch
import os.path as osp
import pickle
import torch.nn.functional as F
import numpy as np
from networks.transformer_net import TransformerNet
import json
import argparse
from detectron.datasets import json_dataset
from detectron.datasets import json_dataset_evaluator
from dataset.base_dataset import BaseDataset

def _encodePngData(arr):
    from PIL import Image
    import io
    import base64

    assert len(arr.shape) == 3, "Expected a 3D array as an input," \
                                " got a {0}D array".format(len(arr.shape))
    assert arr.shape[2] == 3, "Expected first array dimension of size 3," \
                              " got {0}".format(arr.shape[0])
    assert arr.dtype == np.uint8, "Expected an array of type np.uint8, " \
                                  " got {0}".format(arr.dtype)

    im = Image.fromarray(arr)

    buffered = io.BytesIO()
    im.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

parser = argparse.ArgumentParser(description='Ultrapose eval code')
parser.add_argument('--checkpoint', type=str, default='checkpoints/transUltra.pth', help='Path to pretrained model checkpoint')
args = parser.parse_args()

if __name__ == '__main__':
    model_path = args.checkpoint
    if not os.path.exists(model_path):
        print("Invalid checkpoints path. Please check the path or train one first.")
        exit(1)
    save_pkl_path = "checkpoints/transUltra_result.pkl"

    model = TransformerNet(isTrain=False, device='cuda')
    weights = torch.load(model_path, map_location="cpu")
    model.set_weights(weights)
    model = model.cuda()
    model.eval()
    output_img_size = (512, 512)

    image_dir = 'data/ultrapose/val2014'
    ann_path = 'data/ultrapose/annotations/densepose_valminusminival2014.json'
    ann_params = json.load(open(ann_path))

    save_list = []
    with torch.no_grad():
        for idx, item in enumerate(ann_params['annotations']):
            image_path = osp.join(image_dir, ann_params['images'][item['image_id']]['file_name'])
            image_raw = cv2.imread(image_path)
            kpnts17 = np.array(item['keypoints']).reshape(-1, 3)

            (bx, by, bw, bh) = [int(x) for x in item['bbox']]
            bbox = np.array([bx, by, bx+bw-1, by+bh-1], dtype=np.float32)
            cx, cy = bx + bw * 0.5, by + bh * 0.5
            ratio = min(output_img_size[0] / bw, output_img_size[1] / bh) * 0.9
            M_image = cv2.getRotationMatrix2D((cx, cy), 0, ratio)
            M_image[:, 2] += np.array([output_img_size[0]*0.5 - cx,
                                       output_img_size[1]*0.5 - cy])

            M_label = cv2.getRotationMatrix2D((bw*0.5, bh*0.5), 0, ratio)
            M_label[:, 2] += np.array([output_img_size[0]*0.5 - bw*0.5,
                                       output_img_size[1]*0.5 - bh*0.5])

            image = cv2.warpAffine(image_raw, M_image, output_img_size)
            new_bbox = BaseDataset.transform_points(bbox.reshape(-1, 2), M_image).reshape(-1).astype(np.int)
            (nx1, ny1, nx2, ny2) = new_bbox


            kpnts17[:, :] -= np.array([[bx, by, 0]], dtype=np.int)
            kpnts17[:,:2] = BaseDataset.transform_points(kpnts17[:,:2], M_label)

            skl = BaseDataset.draw_connect_keypoints([kpnts17], 512, 512).astype(np.float32)

            input_conditon = np.concatenate([image/255, skl/255], axis=2)
            input = {'input_conditon': torch.from_numpy(input_conditon.transpose([2, 0, 1]).astype(np.float32)).view(1, 6, 512, 512).cuda()}

            ret = model(input)
            s_img = F.interpolate(ret['s_img'][:,:,ny1:ny2+1,nx1:nx2+1], (bh, bw), mode="bilinear", align_corners=False)[0].detach().cpu().numpy()
            i_img = F.interpolate(ret['i_img'][:,:,ny1:ny2+1,nx1:nx2+1], (bh, bw), mode="bilinear", align_corners=False)[0].detach().cpu().numpy()
            u_img = F.interpolate(ret['u_img'][:,:,ny1:ny2+1,nx1:nx2+1], (bh, bw), mode="bilinear", align_corners=False)[0].detach().cpu().numpy()
            v_img = F.interpolate(ret['v_img'][:,:,ny1:ny2+1,nx1:nx2+1], (bh, bw), mode="bilinear", align_corners=False)[0].detach().cpu().numpy()

            ret_iuv = np.zeros((bh, bw, 3), dtype=np.uint8)
            coars_mask = np.argmax(s_img, axis=0)
            I_idx2 = np.argmax(i_img, axis=0)

            for i in range(bh):
                for j in range(bw):
                    if coars_mask[i, j] == 0:
                        continue

                    I = I_idx2[i, j]
                    ret_iuv[i, j, 0] = int(I)
                    ret_iuv[i, j, 1] = int(np.clip(u_img[I, i, j] * 255, 0, 255))
                    ret_iuv[i, j, 2] = int(np.clip(v_img[I, i, j] * 255, 0, 255))


            ret_raw = np.zeros((512, 512, 3), dtype=np.uint8)
            bbox = bbox.astype(int)
            ret_raw[bbox[1]:bbox[3]+1,bbox[0]:bbox[2]+1,:] = ret_iuv[:,:,:]

            show = (image_raw * 0.5 + ret_raw * 0.5).astype(np.uint8)

            tmp_ret = dict(
                category_id=item['category_id'],
                uv_shape=[3, bh, bw],
                image_id=item['image_id'],
                score=1.0,
                bbox=item['bbox'],
                uv=ret_iuv.transpose(2, 0, 1),
            )
            save_list.append(tmp_ret)
            print(idx, len(ann_params['annotations']), image_path)

    pickle.dump(save_list, open(save_pkl_path, 'wb'))

    js_dataset = json_dataset.JsonDataset('ultrapose_valminusminival2014')
    json_dataset_evaluator._do_body_uv_eval(js_dataset, save_pkl_path, ' ')