# UltraPose: Synthesizing Dense Pose with 1 Billion Points by Human-body Decoupling 3D Model
Official repository for the **ICCV 2021** paper:  

**UltraPose: Synthesizing Dense Pose with 1 Billion Points by Human-body Decoupling 3D Model**  [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yan_UltraPose_Synthesizing_Dense_Pose_With_1_Billion_Points_by_Human-Body_ICCV_2021_paper.pdf)]

Haonan Yan, Jiaqi Chen, Xujie Zhang, Shengkai Zhang, Nianhong Jiao, Xiaodan Liang, Tianxiang Zheng
 
The dataset is now available at [Baidu net disk](https://pan.baidu.com/s/13ms66BO_PuWopkfkzZ_IYQ) (code: 477s) and [google drive](https://drive.google.com/file/d/1QyJjFrPrACzFI2fZizIyAcks0mdxieOV/view?usp=sharing).




## Introduction
![teaser](png/fig2.png)
In this work, we introduce a new 3D human-body model with a series of decoupled parameters that could freely control the generation of the body. Furthermore, we build a data generation system based on this decoupling 3D model, and construct an ultra dense synthetic benchmark UltraPose, containing around 1.3 billion corresponding points.

## Installation
We recommend creating a clean [conda](https://docs.conda.io/) environment and install all dependencies.
You can do this as follows:

step1
```
conda create -n ultrapose python=3.7
conda activate ultrapose
```
step2
```
conda install pytorch=1.7.1 torchvision cudatoolkit=10.2 -c pytorch
```
step3
```
pip install ml-collections opencv-python imgaug visdom pycocotools Cython future h5py
```

You need to build python3 densepose for evaluation. You can do this as follows:
```
cd $UltraPoseDir/eval
make
cd $UltraPoseDir/eval/DensePoseData
bash get_eval_data.sh
```

## Training

For single GPU training, please use default configurations by running:

```
python train.py --dataroot data/ultrapose
```
Besides, you can also use visdom to monitor the training process.
```
python -m visdom.server
python train.py --dataroot data/ultrapose --use_visdom
```
For multi-GPU training with default configurations, you can modify `train_transformer.sh` accordingly and run:
```
sh train_transformer.sh
```

## Evaluation
```
python evaluation.py
```

## Dataset
The dataset is now available from [Baidu net disk](https://pan.baidu.com/s/13ms66BO_PuWopkfkzZ_IYQ) (code: 477s) or [google drive](https://drive.google.com/file/d/1QyJjFrPrACzFI2fZizIyAcks0mdxieOV/view?usp=sharing).

Extract the data and put them under `$UltraPoseDir/data`.

|  Dataset   | Persons  | Points  | #Avg Density  | Mask Resolution  | No error |
| :----: | :----: | :----: |:----:  | :----: | :----: |
|  Densepose-COCO  | 49K  |5.2M  |106  |256x256  |  | 
|  UltraPose  | 5K  |13M  |2.6K  |512x512  | ✓ |

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
- [vposer](https://github.com/nghorbani/human_body_prior)
- [densepose](https://github.com/facebookresearch/DensePose)
- [densepose_python3](https://github.com/stimong/densepose_python3)


## Citation
If you use this code or Ultrapose for your research, please cite our work:
```
@inproceedings{yan2021ultrapose,
  title={UltraPose: Synthesizing Dense Pose With 1 Billion Points by Human-Body Decoupling 3D Model},
  author={Yan, Haonan and Chen, Jiaqi and Zhang, Xujie and Zhang, Shengkai and Jiao, Nianhong and Liang, Xiaodan and Zheng, Tianxiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10891--10900},
  year={2021}
}
```
