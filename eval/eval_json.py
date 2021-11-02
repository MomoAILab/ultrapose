from detectron.datasets import json_dataset
from detectron.datasets import json_dataset_evaluator
import json
import cv2


if __name__ == '__main__':
    # res_path = '/home/songkey/workspace/data/densepose-gen/result/yan.haonan/model_FCN_head.test.pkl'
    # res_path = '/home/songkey/workspace/data/densepose-gen/result/dp-model0317/res_dp_depth_ep100_ct8383_bs8_ws6_coco_data_fpn_model.pth.pkl'
    res_path = '/home/songkey/workspace/data/densepose-gen/result/15w/res2_dp_ep133_ct16755_bs10_ws8_big_data_pixel2pixelhd_model.pth.pkl'
    res_path = '/home/songkey/workspace/data/densepose-gen/result/coco-all-result/res2_dp_ep148_ct31905_bs10_ws8_coco_data_pixel2pixelhd_model.pth.pkl'
    res_path = '/home/songkey/workspace/data/densepose-gen/result/15w/res_dp_ep182_ct22880_bs10_ws8_big_data_pixel2pixelhd_model.pth.pkl'
    res_path = '/home/songkey/workspace/data/densepose-gen/result/coco-all-result/res_dp_ep136_ct26025_bs10_ws8_coco_data_pixel2pixelhd_model.pth.pkl'

    res_path = '/home/songkey/workspace/data/densepose-gen/result/coco-all-result/0317-coco-framework/model_FCN_head_15w.test.pkl'
    res_path = '/home/songkey/workspace/data/densepose-gen/result/15w/res_dp_ep273_ct34255_bs10_ws8_big_data_pixel2pixelhd_model.pth.pkl'


    # 2021-10-29
    # res_path = '/media/johnny/5CD86FA3D86F7A62/checkpoints/badDP/ultraTrans/res_dp_ep39_ct13320_bs6_ws2_coco_data_transformer_model.pth.pkl'    # epoch 39
    res_path = "/media/johnny/5CD86FA3D86F7A62/checkpoints/badDP/ultraTrans/DL_15w.pkl"
    res_path = "/media/johnny/5CD86FA3D86F7A62/checkpoints/badDP/ultraTrans/DL_CONF_15w.pkl"
    res_path = "/media/johnny/5CD86FA3D86F7A62/checkpoints/badDP/ultraTrans/FCN_15w.pkl"
    res_path = "/media/johnny/5CD86FA3D86F7A62/checkpoints/badDP/ultraTrans/DL_4k.pkl"          # dl 4k

    # import pickle
    # params = pickle.load(open(res_path, 'rb'))
    # for i in range(len(params)):
    #     print(params[i]['uv'].shape, params[i]['uv_shape'], params[i]['bbox'])
    #     parama = params[i]
    #     cv2.imshow('show', parama['uv'].transpose(1, 2, 0))
    #     cv2.waitKey()
    #     print('hello')

    # js_dataset = json_dataset.JsonDataset('dense_coco_2014_valminusminival')
    # js_dataset = json_dataset.JsonDataset('hn_densepose_val2014')
    js_dataset = json_dataset.JsonDataset('densepose_valminusminival2014')
    # js_dataset = json_dataset.JsonDataset('hn_densepose_train500')
    # js_dataset = json_dataset.JsonDataset('densepose_val5000')
    json_dataset_evaluator._do_body_uv_eval(js_dataset, res_path, ' ')