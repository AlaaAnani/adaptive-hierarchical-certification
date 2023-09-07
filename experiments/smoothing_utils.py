import argparse
import os
import sys

import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

import _init_paths
import models
import datasets
from config import config
from config import update_config
#from core.function import testval, test 
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

from tqdm import tqdm
import scipy.stats as sps
from math import floor
import itertools
import cv2
import path 
import pandas as pd
# directory reach
import copy
from scipy.ndimage.interpolation import shift
from itertools import permutations
from multiclassify_utils import certify, Log, setup_segmentation_args, str2bool
from smoothing_utils import *
import yaml
from scipy.ndimage.filters import generic_filter
import numpy as np
import scipy
from PIL import Image
import glob
from pathlib import Path

def filter_function(invalues):
   invalues_mode = scipy.stats.mode(invalues, axis=None, nan_policy='omit')
   return invalues_mode[0]

def create_acdc_train_test_ls(root_dir, dataset_name):
    os.makedirs(f'{root_dir}/list/{dataset_name}/', exist_ok=True)
    mode = ['train', 'val', 'test']
    for m in mode:
        rgb_ls = glob.glob(f'{root_dir}/{dataset_name}/rgb_anon/*/{m}/*/*.png')
        rgb_ls.extend(list(glob.glob(f'{root_dir}/{dataset_name}/rgb_anon/{m}_ref/*/*.png')))
        rgb_ls = [pth.replace(f'{root_dir}/{dataset_name}/', '') for pth in rgb_ls]


        final_ls = []
        if mode == 'test':
            final_ls = [pth+'\n' for pth in rgb_ls]
        else:
            for i, pth in enumerate(rgb_ls):
                gt_img_name = pth.split('/')[-1].replace('rgb_anon', 'gt_labelIds')
                pth = pth.replace('rgb_anon', 'gt')
                dir = '/'.join(pth.split('/')[:-1])
                gt_pth = f'{dir}/{gt_img_name}'
                if os.path.exists(f'{root_dir}/{dataset_name}/{gt_pth}'):
                    final_ls.append(f'{rgb_ls[i]}\t{gt_pth}\n')

        with open(f'{root_dir}/list/{dataset_name}/{m}.lst', 'w') as f:
            f.writelines(final_ls)

def load_data(config, args):
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]) # (1024, 2048)
    if config.DATASET.DATASET == 'acdc' and len(os.listdir(f'{config.DATASET.ROOT}/list/{config.DATASET.DATASET}')) < 3:
        create_acdc_train_test_ls(config.DATASET.ROOT, config.DATASET.DATASET)
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        center_crop_test=args.crop,
                        downsample_rate=1,
                        normalize=False)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    return test_dataset, testloader

def load_model(config, gpus):
    # build model
    model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)
    model_state_file = config.TEST.MODEL_FILE 
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    return model
    
def smooth_seg_map(array, smooth_filter_size):
    function = lambda array: generic_filter(array, function=filter_function, size=smooth_filter_size)   
    return function(array)

class DotDictify(dict):
    MARKER = object()

    def __init__(self, value=None):
        if value is None:
            pass
        elif isinstance(value, dict):
            for key in value:
                self.__setitem__(key, value[key])
        else:
            raise TypeError('expected dict')

    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDictify):
            value = DotDictify(value)
        super(DotDictify, self).__setitem__(key, value)

    def __getitem__(self, key):
        found = self.get(key, DotDictify.MARKER)
        if found is DotDictify.MARKER:
            found = DotDictify()
            super(DotDictify, self).__setitem__(key, found)
        return found

    __setattr__, __getattr__ = __setitem__, __getitem__

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str, default='/BS/mlcysec/work/robust-segmentation/code/hrnet_seg/tools/test.yml')

    setup_segmentation_args(parser)
    parser.add_argument('--crop', type=str2bool, default=False)
    parser.add_argument('--unscaled', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('-N0', type=int, default=0)
    parser.add_argument('-NN', type=int, nargs='+', default=[])
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    with open(args.cfg) as f:
        conf = yaml.safe_load(f)
    return args, DotDictify(conf)

def get_confusion_matrix(label, seg_pred, size, num_class, ignore=-1):
    """
    Cal cute the confusion matrix by given label and pred
    """
    seg_gt = np.asarray(
    label.cpu().numpy(), dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix
def save_pvals(pvals, path):
    np.save(f'{path}.npy', pvals)
    cv2.imwrite(f'{path}.png', pvals[0]*255)
    cv2.imwrite(f'{path}_log.png', np.log(pvals[0]))

def save_segmentation(seg, test_dataset, palette, path, abstain, abstain_mapping):
    img = np.asarray(seg[0, ...], dtype=np.uint8)
    I = (img == abstain)
    img = test_dataset.convert_label(img, inverse=True) # converts to id
    img[I] = abstain_mapping
    img = Image.fromarray(img)
    img.putpalette(palette)
    img.save(path)

def sample(image_np01, size, n, sigma, model, config, test_dataset, do_tqdm=True, unscaled=False):
    BS = config.TEST.BATCH_SIZE_PER_GPU
    out, out_pred = [], []
    remaining = n
    if do_tqdm: pbar = tqdm(total=n)
    with torch.no_grad():
        while (remaining) > 0:
            cnt = min(remaining, BS)
            pred = test_dataset.multi_scale_inference_noisybatch(
                        model, 
                        cnt, sigma,
                        image_np01,
                        normalize=True,
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST,
                        unscaled=unscaled) # torch.Size([1, 19, 1024, 2048])
            if not unscaled and (pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]):
                pred = F.interpolate(pred, (size[-2], size[-1]), 
                                   mode='bilinear')
            out_pred.append(pred.cpu().numpy())
            pred = pred.argmax(dim=1).cpu().numpy() # (1, 1024, 2048)
            out.append(pred)
            remaining -= cnt
            if do_tqdm: pbar.update(cnt)
    if do_tqdm: pbar.close()
    return np.concatenate(out), np.concatenate(out_pred)


def get_slice(n):
    if n == 0: return ':'
    if n > 0: return '0:n'
    else: return '-n:'


def fast_shift(x, shift_value, mode='same'):
    x_ = x
    num_axes = len(x.shape)
    for axis, axis_shift in enumerate(shift_value):
        if axis_shift == 0:continue
        x_ = np.roll(x_, axis_shift, axis=axis)
        sl = get_slice(axis_shift).replace('n', 'axis_shift')
        sl = ':,' * axis + f'{sl}' +', :'*(num_axes - axis -1)
        exec(f'x_[{sl}] = x[{sl}]')
    return x_
        


def create_shifted_maps(x, ignore=255, window_size=3):
    shift_indices = np.arange(int(window_size/2.0)+1)
    shift_indices = set(list(shift_indices) + list(-shift_indices))
    shift_indices = list(set([p for p in itertools.product(shift_indices, repeat=len(x.shape))]))
    if len(x.shape) == 3:
        shift_indices = list(set([(0, si[1], si[2]) for si in shift_indices]))
    shifted_ls = []
    for shift_value in shift_indices:
        if len(shift_value) > 2:
            shift_value = (0, shift_value[1], shift_value[2])
        shifted_ls.append(fast_shift(x, shift_value, mode='nearest'))
    if len(x.shape) == 2:
        return np.stack(shifted_ls)
    return np.concatenate(shifted_ls)

def relaxed_compare(label, pred, ignore=255, window_size=3, abstain=None):

    label_ = label[0]
    pred_ = pred[0]
    tp = 0
    bin_map = np.bool8(np.zeros(label_.shape))
    shifted_ls = create_shifted_maps(label_, window_size)
    for shifted_label in shifted_ls:
        comp_map = (shifted_label == pred_)
        bin_map = bin_map | comp_map

    if abstain is not None:
        bin_map = bin_map[(label_ != ignore) & (pred_ != abstain)]
    else:
        bin_map = bin_map[label_ != ignore]
    # TODO: make a confusion matrix instead of a binary map
    tp, num_pixels = bin_map.sum(), len(bin_map)
    return tp/num_pixels


def to_hierarchical(labels, abstain=None):
    lookup_table = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 6, 6]+ [255]*(255-18)
    if torch.is_tensor(labels):
        lookup_table = torch.tensor(lookup_table)
        labels = labels.long()
    else:
        lookup_table = np.array(lookup_table)
        labels = labels.astype(np.longlong)
    lookup_table[-1] = 255
    new_labels = []
    for label in labels:
        new_label = lookup_table[label]
        if abstain:
            abstain_idx = label == abstain
            new_label[abstain_idx] = abstain
        new_labels.append(new_label)
    if torch.is_tensor(labels):
        return torch.stack(new_labels)
    else:
        return np.stack(new_labels)

def get_edgemap(label, num_classes, ignore):
    label_ = label
    label_[label == ignore] = num_classes
    label_ = np.uint8(label_/label_.max()*255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edge_map = np.uint8(cv2.dilate(cv2.Canny(label_[0], 0, 100), kernel))
    return np.uint8(edge_map/255.0)

def get_histogram(edge_map, label, classes_certify, abstain, num_classes, ignore=255):
    # get histogram for boundary vs. non-boundary
    abstained_edge_map = edge_map[(classes_certify[0] == abstain) & (label[0] != ignore)]
    num_boundary, num_non_boundary = ((edge_map[label[0] != ignore])==1).sum(), ((edge_map[label[0] != ignore])==0).sum()
    boundary_histogram = [len(abstained_edge_map) - abstained_edge_map.sum(), abstained_edge_map.sum()]
    boundary_histogram[0] = float(boundary_histogram[0])/num_non_boundary
    boundary_histogram[1] = float(boundary_histogram[1])/num_boundary
    # get histogram for classes
    abstained_labels = label[classes_certify == abstain] 
    abstained_labels = abstained_labels[abstained_labels != ignore]

    return np.bincount(abstained_labels, minlength=num_classes-1), boundary_histogram