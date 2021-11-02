"""This module contains simple helper functions """
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

def rgb_to_binary(input: torch.Tensor) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 0:
        raise ValueError("Input size must have a shape of (H, W). Got {}"
                         .format(input.shape))

    v = torch.chunk(input.unsqueeze(dim=-1), chunks=1, dim=-1)
    # binary: torch.Tensor = torch.true_divide(v[0], v[0] + 1e-6)
    binary: torch.Tensor = torch.div(v[0], v[0] + 1e-6)

    return binary

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def transfor_weights(dst, weights):
    model_dict = dst.state_dict()
    pretrained_dict = {}
    for ix, (k, v) in enumerate(model_dict.items()):
        tmp_k = k
        # tmp_k = 'module.encoder.' + k
        if k in weights and weights[k].data.shape == v.data.shape:
            pretrained_dict[tmp_k] = weights[k]
        else:
            print('ignore {}'.format(k))

    model_dict.update(pretrained_dict)
    dst.load_state_dict(model_dict)

def load_weights(model, weights_path):
    weights = torch.load(weights_path, map_location='cpu')  # ['model']
    transfor_weights(model, weights)
    return model


def resample_fine_and_coarse_segm_tensors_to_bbox(
    fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs: tuple
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = F.interpolate(
        coarse_segm, (h, w), mode="bilinear", align_corners=False
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
        F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
        * (coarse_segm_bbox > 0).long()
    )
    return labels
