# %%
from glob import glob
import cv2
import os
import pandas as pd
import numpy as np
from graph_utils import *
from tqdm import tqdm

log_dir = '/BS/mlcysec/work/robust-segmentation/code/hrnet_seg/logs'
to_title_dict = {'non_abstain': 'Percentage of Certified Pixels', 
                'n': 'Sampling Rate (n)', 
                'accuracy': 'Accuracy',
                'time_sample': 'Sampling Time (s)', 
                'time_cert_overall': 'Certification Time (s)',
                'relaxed_pixel_acc': 'Relaxed Pixel Accuracy'}
timing_params = ['time_sample', 'time_shift', 'time_pvals', 'time_correction']
main_params = ['n', 'n0', 'tau']
acc_params = ['accuracy', 'relaxed_pixel_acc']
corr = ['holm', 'bonferroni']
param1s = ['non_abstain', 'accuracy', 'relaxed_pixel_acc']
setups = ['baseline', 'hie', 'relax', 'relax_hie'] 

#exp_dirs = [f'{log_dir}/cityscapes_final_{setup}' for setup in setups]
setups = ['baseline_prompt', 'smoothed_7']
exp_dirs = [f'{log_dir}/cityscapes_{setup}' for setup in setups]
for exp in tqdm(exp_dirs):
    for param in ['n']:
        graph_gifs(exp, param=param)
    
#     for param1 in param1s:
#         graph_param1_vs_param2(exp, param1, 'n', to_title_dict)
#     graph_bar_params_vs_param1(exp, timing_params, 'n', to_title_dict, y_title = 'Time', corr=corr)
#     graph_histogram(exp, 'n')

#graph_boundary_histogram_for_all(setups[1:], exp_dirs)
# graph_confusion_matrix(log_dir, params = ['n', 'n0', 'tau'])
 # %%
out_pred = np.load('/BS/mlcysec/work/robust-segmentation/code/hrnet_seg/logs/0_sample_outpred_1000.npy') # (1, 1000, 19, 1024, 2048)