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
from PIL import Image


import _init_paths
import models
import datasets
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
from mode_filter import ModePool2d
import pickle
# setting path

def test_wrapper(args, config, test_dataset, testloader, model,
                hierarchical=True, relaxation=False,  window_size=3, 
                smoothing=False, smooth_kernel_size=5,
                log_name=None, create_df=True):
                

    images_stats_dict = {}
    STATS_COLS = ['idx', 'smoothed', 'smooth_kernel_size', 'hierarchical', 'n', 'n0', 'N',
            'tau', 'alpha', 'sigma', 'radius', 'baseline', 'non_abstain',
            'accuracy', 'IoU', 'mean_accuracy', 'mean_iou', 'correction',  
            'name', 'scale', 'batch_size', 'model',
            'time_baseline', 'time_sample', 'time_pvals', 'time_correction', 
            'time_cert_overall', 'confusion_matrix', 
            'abstain_histogram', 'abstain_histogram_boundary_normalized', 
            'relaxation', 'relax_window_size', 'time_shift', 'relaxed_pixel_acc', 
            'num_voters',  'flip']
    model.eval()

    log = Log(config.DATASET.DATASET + args.name, log_name=log_name)

    df_ls_file_name = f'{str(log.folder())}/df_ls.pkl'
    if create_df:
        df_ls = pickle.load(open(df_ls_file_name, 'rb'))
        pd.DataFrame(df_ls, columns=STATS_COLS).to_csv(f'{str(log.folder())}/images_df.csv', index=False)
        print("Created images_df successfully!")
        return
        
    try: 
        df_ls = pickle.load(open(df_ls_file_name, 'rb'))
    except:
        df_ls = []
    log.write(str(args) + "\n")

    with open(log.folder() / "config.cfg", 'w') as f:
        f.write(str(config))
    


    if hierarchical:
        num_classes = 7+1
    else:
        num_classes = config.DATASET.NUM_CLASSES+1 # threat abstain as an additional class #19+1

    abstain = num_classes - 1 # label 8 if hierarchical else label 19
    abstain_mapping = 254 
    canny = False
    corr = ['bonferroni', 'holm']
    out_confusion_matrix = {}
    out_confusion_matrix['baseline'] = np.zeros((num_classes, num_classes))
    outname_base = "certify" if not args.unscaled else "certify_unscale"
    for k in range(len(args.n)):
        tau = args.tau[k] 
        n = args.n[k]
        n0 = args.n0[k]
        for c in corr:
            out_confusion_matrix[f"{outname_base}_{c}_{n}_{n0}_{tau}"] = np.zeros((num_classes, num_classes)) # dict_keys(['baseline', 'certify_bonferroni_1000_0.75', 'certify_holm_1000_0.75'])

    out_confusion_matrix['baseline'] = np.zeros((num_classes, num_classes))
    palette = test_dataset.get_palette(256)
    palette[abstain_mapping * 3 + 0] = 255
    palette[abstain_mapping * 3 + 1] = 255
    palette[abstain_mapping * 3 + 2] = 255

    m = floor(len(testloader) / args.N) # 500/50 = 10
    itt = itertools.islice(testloader, 0, None, m)
    itt = enumerate(itertools.islice(itt, args.N))
    base_log_dir = '/BS/mlcysec/work/robust-segmentation/code/hrnet_seg/logs/'
    with torch.no_grad():
        for index, batch in tqdm(itt, desc='Main Loop'): #

            if index < args.N0: continue
            if len(args.NN) > 0 and index not in args.NN: continue
            print("Image", index, file=log)
            image, label, _, name = batch # image=torch.Size([1, 3, 1024, 2048]), label=torch.Size([1, 1024, 2048]), name=['frankfurt_000000_000...e_labelIds']

            per_image_stats_row = {key: None for key in STATS_COLS}
            per_image_stats_row['smoothed'] = smoothing
            per_image_stats_row['smooth_kernel_size'] = smooth_kernel_size
            per_image_stats_row['relaxation'] = relaxation
            per_image_stats_row['relax_window_size'] = window_size
            per_image_stats_row['hierarchical'] = hierarchical
            per_image_stats_row['idx'] = index
            per_image_stats_row['scale'] = config.TEST.SCALE_LIST
            per_image_stats_row['flip'] = {}#config.TEST.FLIP_TEST
            per_image_stats_row['batch_size'] = config.TEST.BATCH_SIZE_PER_GPU
            per_image_stats_row['model'] = config.TEST.MODEL_FILE
            per_image_stats_row['N'] = args.N
            per_image_stats_row['alpha'] = args.alpha
            per_image_stats_row['sigma'] = args.sigma
            per_image_stats_row['name'] = name

            image_np = image.numpy()[0].transpose((1, 2, 0)).astype(np.float32).copy() # (1024, 2048, 3)
            image_np01 = image_np/255.0
            size = label.size() # torch.Size([1, 1024, 2048])
            
            # save image and label
            img = Image.fromarray(image_np.astype(np.uint8))
            img.save(log.folder() / f"{index}_image.png")
            if hierarchical:
                label = to_hierarchical(label)
                
            save_segmentation(label, test_dataset, palette, log.folder() / f"{index}_gt.png", abstain, abstain_mapping )
            del img

            edge_map = get_edgemap(copy.deepcopy(label).cpu().numpy(), num_classes, config.TRAIN.IGNORE_LABEL)
            cv2.imwrite(f'{log.folder()}/{index}_canny.png', edge_map*255)

            t0 = time.time()
            classes_baseline, out_pred = sample(image_np01, size, 1, 0, model, config, test_dataset, do_tqdm=False) # (1, 1024, 2048)
            pickle.dump(out_pred, open(f'{base_log_dir}/{index}_baseline_outpred.pkl', 'wb'))
            t1 = time.time()
            time_baseline = t1-t0

            if hierarchical: classes_baseline = to_hierarchical(classes_baseline)
            if smoothing: classes_baseline =  ModePool2d(kernel_size=smooth_kernel_size)(torch.tensor(classes_baseline).unsqueeze_(0)).squeeze_(0).cpu().numpy()

            save_segmentation(classes_baseline, test_dataset, palette, log.folder() / f"{index}_baseline.png", abstain, abstain_mapping)

            if relaxation:
                relaxed_pixel_acc = relaxed_compare(label.cpu().numpy(), classes_baseline, ignore=config.TRAIN.IGNORE_LABEL, window_size=window_size)
                per_image_stats_row['relaxed_pixel_acc'] = relaxed_pixel_acc

            confusion_matrix = get_confusion_matrix(label, classes_baseline, size, num_classes, config.TRAIN.IGNORE_LABEL)
            pos = confusion_matrix.sum(1)
            res = confusion_matrix.sum(0)
            tp = np.diag(confusion_matrix)
            pixel_acc = tp.sum()/pos.sum()
            mean_acc = (tp/np.maximum(1.0, pos)).mean()
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()

            print('baseline', file=log)
            print('accuracy', pixel_acc, file=log)
            print('IoU', IoU_array, file=log)
            print('time', time_baseline, file=log)
            print(file=log)

            per_image_stats_row['baseline'] = True
            per_image_stats_row['accuracy']  = pixel_acc
            per_image_stats_row['IoU'] = IoU_array
            per_image_stats_row['time_baseline'] = time_baseline
            per_image_stats_row['confusion_matrix'] = confusion_matrix

            df_ls.append(list(per_image_stats_row.values()))

            out_confusion_matrix["baseline"] += confusion_matrix

            if args.baseline: continue

            first_sample = True

            for tau, n, n0 in zip(args.tau, args.n, args.n0):

                t0 = time.time()
                
                if first_sample:
                    try:
                        s = np.load(f'{base_log_dir}/{index}_sample_1000.npy')[:n0+n]
                        raise RuntimeError
                    except:
                        s, out_pred = sample(image_np01, size, 100, args.sigma, model, config, test_dataset, do_tqdm=True, unscaled=args.unscaled)
                        pickle.dump(out_pred, open(f'{base_log_dir}/{index}_sample_outpred_100.pkl', 'wb'))
#                        np.save(f'{base_log_dir}/{index}_sample_1000.npy', s)
#                        s = s[:n0+n]
                    if hierarchical: s = to_hierarchical(s)
                    if smoothing: 
                        try:
                            s = np.load(f'{base_log_dir}/{index}_sample_smoothed_1000.npy')[:n0+n]
                        except:
                            bs = 16
                            s_i = []
                            for i in tqdm(range(0, s.shape[0], bs), desc='ModePool2D'):
                                s_i.append(ModePool2d(kernel_size=smooth_kernel_size)(torch.tensor(s[i:i+bs]).unsqueeze_(1)).squeeze_(1).cpu().numpy())
                            s = np.concatenate(s_i)
                            np.save(f'{base_log_dir}/{index}_sample_smoothed_1000.npy', s)


                    #first_sample = False
                else:
                    s = np.reshape(s, s_shape)  # (N0+N, 1024, 2048)
                    s = s[:n0+n]
      
                s_shape = s.shape # (1100=N0+N, 1024, 2048)
                num_voters = n+n0

                if relaxation:
                    t0 = time.time()
                    s = create_shifted_maps(s, ignore=config.TRAIN.IGNORE_LABEL, window_size=window_size)
                    num_voters = s.shape[0]
                    per_image_stats_row['num_voters'] = num_voters
                    t1 = time.time()
                    time_shift = t1 - t0
                    per_image_stats_row['time_shift'] = time_shift

                s = np.reshape(s, (num_voters, -1)) # (1100, 2097152)
                t1 = time.time()
                time_sample = t1 - t0



                for c in corr:
                    t0 = time.time()
                    classes_certify, pvals, radius, timings = certify(num_classes-1, s, n0, num_voters-n0, args.sigma, tau, args.alpha, abstain=abstain, parallel=True, correction=c) 
                    t1 = time.time()
                    time_cert_overall = t1 - t0

                    classes_certify = np.reshape(classes_certify, (1, *s_shape[1:]) ) #(1, 1024, 2048)
                    pvals = np.reshape(pvals, (1, *s_shape[1:]))

                    save_pvals(pvals, f"{str(log.folder())}/{index}_pvals_{c}_{n}_{n0}_{tau}")
                    
                    if args.unscaled:
                        classes_certify = cv2.resize(np.transpose(classes_certify, (1, 2, 0)).astype(np.float32), dsize=(size[-2], size[-1]), interpolation=cv2.INTER_NEAREST)
                        classes_certify = np.reshape(classes_certify, (1, size[-2], size[-1])).astype(np.int64)
                    save_segmentation(classes_certify, test_dataset, palette, log.folder() / f"{index}_certify_{c}_{n}_{n0}_{tau}.png", abstain, abstain_mapping)
                    time_pvals, time_correction = timings

                    confusion_matrix = get_confusion_matrix(label, classes_certify, size, num_classes, config.TRAIN.IGNORE_LABEL)


                    histogram, abstain_histogram_boundary_normalized = get_histogram(edge_map, label.cpu().numpy(), classes_certify, abstain, num_classes, config.TRAIN.IGNORE_LABEL)
                    relaxed_pixel_acc = None
                    if relaxation:
                        relaxed_pixel_acc = relaxed_compare(label.cpu().numpy(), classes_certify, ignore=config.TRAIN.IGNORE_LABEL, window_size=window_size)

                    out_confusion_matrix[f"{outname_base}_{c}_{n}_{n0}_{tau}"] += confusion_matrix

                    print(f"{outname_base}_{c}_{n}_{n0}_{tau}", file=log)
                    pos = confusion_matrix.sum(1)
                    res = confusion_matrix.sum(0)
                    tp = np.diag(confusion_matrix)
                    pixel_acc = tp.sum()/pos.sum()
                    mean_acc = (tp/np.maximum(1.0, pos)).mean()
                    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                    mean_IoU = IoU_array.mean()

                    I = (classes_certify != abstain)
                    cnt_nonabstain = np.sum(I)
                    print('accuracy', pixel_acc, file=log)
                    print('non_abstain', cnt_nonabstain/classes_certify.size, file=log)
                    print('IoU', IoU_array, file=log)
                    print('time', time_sample, time_pvals, time_correction, file=log)
                    print(file=log)

                    per_image_stats_row['radius'] = radius
                    per_image_stats_row['N'] = n0+n
                    per_image_stats_row['confusion_matrix'] = confusion_matrix
                    per_image_stats_row['relaxed_pixel_acc'] = relaxed_pixel_acc
                    per_image_stats_row['abstain_histogram_boundary_normalized'] = abstain_histogram_boundary_normalized
                    per_image_stats_row['abstain_histogram'] = histogram
                    per_image_stats_row['baseline'] = False
                    per_image_stats_row['n0'] = n0
                    per_image_stats_row['n'] = n
                    per_image_stats_row['tau'] = tau
                    per_image_stats_row['correction'] = c
                    per_image_stats_row['time_baseline'] = time_baseline
                    per_image_stats_row['time_sample'] = time_sample
                    per_image_stats_row['time_pvals'] = time_pvals
                    per_image_stats_row['time_correction'] = time_correction
                    per_image_stats_row['time_cert_overall'] = time_cert_overall
                    per_image_stats_row['accuracy']  = pixel_acc
                    per_image_stats_row['non_abstain'] = cnt_nonabstain/classes_certify.size
                    per_image_stats_row['IoU'] = IoU_array

                    #images_df = images_df.append(per_image_stats_row, ignore_index=True)
                    df_ls.append(list(per_image_stats_row.values()))

                    
                    pickle.dump(df_ls, open(df_ls_file_name, 'wb'))

                    #images_df.to_csv(f'{str(log.folder())}/images_df.csv', index=False)
            print(file=log)

        print('done', file=log)
        print(file=log)
        
        df_ls.append(list(per_image_stats_row.values()))
        pd.DataFrame(df_ls, columns=STATS_COLS).to_csv(f'{str(log.folder())}/images_df.csv', index=False)
        print("Saved results to a dataframe!")
        for key in out_confusion_matrix.keys():
            print(key, file=log)
            pos = out_confusion_matrix[key].sum(1)
            res = out_confusion_matrix[key].sum(0)
            tp = np.diag(out_confusion_matrix[key])
            pixel_acc = tp.sum()/pos.sum()
            mean_acc = (tp/np.maximum(1.0, pos)).mean()
            IoU_array = (tp / np.maximum(1.0, pos + res - tp))
            mean_IoU = IoU_array.mean()

            print('accuracy', pixel_acc, file=log)
            print('mean-accuracy', mean_acc, file=log)
            print('IoU', IoU_array, file=log)
            print('Mean IoU', mean_IoU, file=log)
            print(file=log)


def main():
    args, config = parse_args()
    if args.unscaled: assert len(config.TEST.SCALE_LIST) == 1

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = [i for i in range(torch.cuda.device_count())]
    print('Using gpus', gpus)

    model = load_model(config, gpus)

    test_dataset, testloader = load_data(config, args)


    args.n = [100]
    args.tau = [0.75]
    args.n0 = [10]

    args.alpha = 0.001 #  0.001 error probability (confidence is 1 - alpha)
    args.sigma = 0.25 # basemodel is trained on 0.25
    args.N = 4 # number of images
    args.N0 = -1

    hierarchical = False
    relaxation = False
    smoothing = False
    window_size = 5
    smooth_kernel_size = 7
    #log_name = f'smoothed_{smooth_kernel_size}'
    log_name = 'posteriors'
    create_df = False
    test_wrapper(args, config, test_dataset, testloader,
                model, hierarchical, relaxation, window_size,
                smoothing, smooth_kernel_size, log_name, create_df)

if __name__ == '__main__':
    main()
