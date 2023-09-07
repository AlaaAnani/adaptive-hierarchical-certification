import argparse
import yaml
import torch.backends.cudnn as cudnn
import torch
from smoothing_utils import DotDictify, \
                        create_acdc_train_test_ls, \
                        to_hierarchical, sample, get_confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import shutil
from configargparse import ArgumentParser as ConfigArgumentParser
import torch.nn as nn

from argparse import ArgumentParser
import torchvision

import _init_paths
import models
import datasets
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
from mode_filter import ModePool2d
import pickle
import time
import multiprocessing as mp
import scipy.stats as sps
from scipy import stats
from statsmodels.stats.multitest import multipletests
import cv2
import plotly.graph_objects as go
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import io
import copy
import collections
from twosample import binom_test
from scipy import stats
import matplotlib.pyplot as plt
# adaptive certify utils
from torch.utils.data.distributed import DistributedSampler
from os.path import dirname, abspath

os.chdir(dirname(dirname(abspath(__file__))))
print('working dir is', dirname(dirname(abspath(__file__))))
def mode(sample):
    # for every sample in a pixel 
    c = stats.mode(sample)[0][0]
    cnt = np.sum(sample == c) 
    return c, cnt

def map_test(args_):
    sample, n, tau = args_ # arguments a tuple to be compatible with multiprocessing map
    c, cnt = mode(sample) # mode class, frequency of mode
    return c, sps.binom_test(cnt, n, tau, alternative='greater'), cnt # perform a test s.t. probability of success is tau

def get_mode_and_count_with_ignore(args):
    sample, n, tau, thresh, max_size, num_classes = args # arguments a tuple to be compatible with multiprocessing map

    freq_dict = {}
    certified = False
    modes = np.zeros(num_classes)
    counts = 0
    for val in sample:
        freq_dict[val] = freq_dict.get(val, 0) + 1

    set_size = 0
    while set_size < max_size:
        try:
            mode = max(freq_dict, key=freq_dict.get)
        except:
            break
        counts += freq_dict[mode]
        modes[mode] = 1
        freq_dict.pop(mode)
        pval = sps.binom_test(counts, n, tau, alternative='greater')
        certified = pval <= thresh
        set_size += 1
    if not certified:
        modes[-1:] = 1
        modes[:-1] = 0
    return modes, pval, counts

def mode(sample, n0, num_classes):
    counts = np.zeros((num_classes))
    for i in range(n0):
        counts[sample[i]] += 1
    c = np.argmax(counts)
    cnt = np.sum(sample[n0:] == c) 
    return c, cnt

def map_test(args):
    sample, n0, n, num_classes, tau = args # arguments a tuple to be compatible with multiprocessing map
    c, cnt = mode(sample, n0, num_classes) # mode class, frequency of mode
    return c, sps.binom_test(cnt, n, tau, alternative='greater') # perform a test s.t. probability of success is tau

def pixel_test(args):
    sample, n0, n, tau, thresh, lookup_tables = args
    certified = False

    h_i = 0
    num_classes = 20
    while h_i <= len(lookup_tables):
        if h_i == 0:
            sample_h = sample
        else:
            sample_h = lookup_tables[h_i-1][sample]
        c, cnt = mode(sample_h, n0, num_classes) # mode class, frequency of mode
        pval = sps.binom_test(cnt, n-n0, tau, alternative='greater')
        certified = pval <= thresh
        h_i += 1
        if certified:
            break
    if not certified:
        c = 19
    return c, h_i-1

class Certifier:
    def __init__(self, rank, world_size, jobid, numjobs):
        self.rank, self.world_size, self.jobid = rank, world_size, int(jobid)
        self.numjobs = int(numjobs)
        self.parse_args()
        self.init_model()
        self.init_datasets()
        self.init_writer()
        self.init_pre_certify()
        
        #self.test_calibration()

        self.main_skeleton()

    def parse_args(self):
        parser = ArgumentParser()
        parser = Certifier.add_argparse_args(parser)
        parser = ConfigArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--config", default='configs/cityscapes.ini', is_config_file=True)
        args, _ = parser.parse_known_args()
        with open(args.cfg) as f:
            conf = yaml.safe_load(f)

        config = DotDictify(conf)
        args_dict = vars(args)

        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED

        self.config = config
        self.args = args
        self.args_dict = args_dict

        self.default_log_dir = args.default_log_dir
        self.batch_size = args.batch_size
        self.gpus = args.gpus # [i for i in range(torch.cuda.device_count())]
        self.cdevice=torch.device(f'cuda:{self.rank}')

        self.baseline = args.baseline
        self.certify = args.certify
        self.hierarchical = args.hierarchical
        self.relaxation = args.relaxation
        self.smoothing = args.smoothing
        self.smooth_kernel_size = 7
        self.corr = args.corr
        self.sigma = args.sigma
        self.alpha = args.alpha
        self.tau = args.tau
        self.n = args.n
        self.max_n = args.max_n
        self.greedy = args.greedy
        self.max_sizes = args.max_sizes

    @classmethod
    def add_argparse_args(cls, parser):
        """
        Adds dataset specific parameters to parser
        :param parser:
        :return:
        """
        parser = argparse.ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--N', type=int, default=10) # only run over 50 images
        parser.add_argument('--NN', type=int, default=[], nargs='+') # only run over 50 images
        parser.add_argument('--N0', type=int, default=-1) # start idx
        parser.add_argument('--sigma', type=float, default=0.25)
        parser.add_argument('--alpha', type=float, default=0.001)
        parser.add_argument('--n', type=int, default=[100], nargs='+')
        parser.add_argument('--n0', type=int, default=[10], nargs='+')
        parser.add_argument('--tau', type=float, default=0.75)
        parser.add_argument('--baseline', action="store_true") # run baseline only
        parser.add_argument('--default_log_dir', type=str, default= '')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--smooth_kernel_size', default=7)
        parser.add_argument('--hierarchical', action="store_true")
        parser.add_argument('--relaxation', action="store_true")
        parser.add_argument('--smoothing', action="store_true")
        parser.add_argument('--window_size', type=int, default=5)
        parser.add_argument('--cfg', type=str, default='configs/cityscapes.yml')
        parser.add_argument('--crop', action="store_true")
        parser.add_argument('--unscaled', action="store_true")
        parser.add_argument('--name', type=str, default='')
        parser.add_argument('--corr', type=str, nargs='+')
        parser.add_argument('--max_n', type=int, default=300)
        parser.add_argument('--gpus', type=int, default=[1], nargs='+')
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--certify', action="store_true")
        parser.add_argument('--greedy', action="store_true")
        parser.add_argument('--max_sizes', type=int, default=[1], nargs='+')

        return parser
    
    def init_model(self):
        self.log('Initializing model')
        self.model = self.load_model(self.config)


    def init_datasets(self):
        self.log('Initializing datasets')
        if self.config.DATASET.DATASET == 'acdc' and len(os.listdir(f'{self.config.DATASET.ROOT}/list/{self.config.DATASET.DATASET}')) < 3:
            create_acdc_train_test_ls(self.config.DATASET.ROOT, self.config.DATASET.DATASET)
        self.test_dataset, testloader = self.load_data(self.config, self.args)
        self.test_dataset = self.test_dataset
        self.testloader = testloader

        self.labels_dict_h = {0:'drivable', 1:'flat non-drivable', 2:'obstacle', 3:'road sign', 4:'sky', 5:'human', 6:'vehicle', 7:'abstain'}
        self.labels_dict = pickle.load(open('data/cityscapes_trainIds.pkl', 'rb'))
        self.labels_dict_h = {0:'drivable', 1:'flat non-drivable', 2:'obstacle', 3:'road sign', 4:'sky', 5:'human', 6:'vehicle', 7:'abstain'}
        self.labels_dict = pickle.load(open('data/cityscapes_trainIds.pkl', 'rb'))
        self.labels_dict[19] = 'abstain'

        self.labels_dict_h1 = self.labels_dict # 19
        self.labels_dict_h2 = {0:'drivable', 1:'sky', 2:'construction & vegetation', 3:'traffic sign', 4:'human', 5:'vehicle', 6:'flat obstacle', 7:'abstain'} # 7
        self.labels_dict_h3 = {0: 'drivable', 1:'static obstacle', 2: 'dynamic obstacle', 3:'flat obstacle', 4: 'abstain'} # 5
        self.labels_dict_h4 = {0: 'drivable', 1:'obstacle', 2:'abstain'} # 3
        self.lookup_tables = [np.array([0, 6, 2, 2, 2, 2, 3, 3, 2, 6, 1, 4, 4, 5, 5, 5, 5, 5, 5]),
                              np.array([0, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2]),
                              np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])]
        self.lookup_info_gain = [np.array([1]*19),
                                np.array([1, 1, 5, 2, 2, 6, 2]),
                                np.array([1, 8, 8, 2]),
                                np.array([1, 18])]
        def linear_n(n):
            return [(300/n/12)*i for i in range(1, 4)]
        def exp_n(n):
            return [(300/n*0.25)**(4-i) for i in range(1, 4)]
        self.threshold_functions = {'linear': [0.25* i for i in range(1, 4)],
        'exp_0.75': [0.75**(4-i) for i in range(1, 4)],
        'exp_0.40': [0.4**(4-i) for i in range(1, 4)],
        'linear_n': linear_n,
        'exp_n': exp_n}
        self.labels_dicts = [self.labels_dict_h2, self.labels_dict_h3, self.labels_dict_h4]

    def init_writer(self):
        self.log('Initializing writer')
        os.makedirs(self.default_log_dir, exist_ok=True)
        # experiment_folder = f'{self.default_log_dir}/version_{len(os.listdir(self.default_log_dir))+100}'
        # #os.makedirs(experiment_folder, exist_ok=True)
        # #print(experiment_folder)
        # #shutil.copyfile(self.args.config, f'{experiment_folder}/config.ini')
        # self.writer = SummaryWriter(experiment_folder)

    def load_model(self, config):
        # build model
        model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)
        model.to(self.cdevice)
        model_state_file = config.TEST.MODEL_FILE 
        pretrained_dict = torch.load(model_state_file, map_location=self.cdevice)#, map_location=self.cdevice)
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model = nn.DataParallel(model, device_ids=self.gpus).cuda()
        return model
    
    def load_data(self, config, args):
        test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]) # (1024, 2048)
        test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                            root=config.DATASET.ROOT,
                            list_path=config.DATASET.TEST_SET,
                            num_samples=100,
                            num_classes=config.DATASET.NUM_CLASSES,
                            multi_scale=False,
                            flip=False,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            base_size=config.TEST.BASE_SIZE,
                            crop_size=test_size,
                            center_crop_test=args.crop,
                            downsample_rate=1,
                            normalize=False,
                            jobid=self.jobid,
                            numjobs=self.numjobs)
        
        datasampler = DistributedSampler(
            dataset=test_dataset,
            rank=self.rank,
            num_replicas=self.world_size)
        
        testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=datasampler,
            pin_memory=True)
        return test_dataset, testloader

    def init_pre_certify(self):
        self.num_classes = 7+1 if self.hierarchical else self.config.DATASET.NUM_CLASSES+1 # 19+1
        self.abstain_label = self.num_classes - 1
        self.abstain_mapping = 254

        self.palette = self.test_dataset.get_palette(256)
        self.palette[self.abstain_mapping * 3 + 0] = 255
        self.palette[self.abstain_mapping * 3 + 1] = 255
        self.palette[self.abstain_mapping * 3 + 2] = 255
    
    def sample(self, image_np01, size, n, sigma, posteriors=False, do_tqdm=False, cuda_id=None):
        print('sample')
        BS = self.config.TEST.BATCH_SIZE_PER_GPU
        # print('Sampling @ batch size =', BS)
        out, out_pred = [], []
        remaining = n
        if do_tqdm: pbar = tqdm(total=n)
        with torch.no_grad():
            while (remaining) > 0:
                cnt = min(remaining, BS)
                pred = self.test_dataset.multi_scale_inference_noisybatch(
                            self.model, 
                            cnt, sigma,
                            image_np01,
                            normalize=True,
                            scales=self.config.TEST.SCALE_LIST, 
                            flip=self.config.TEST.FLIP_TEST,
                            unscaled=False,
                            cuda_id=cuda_id) # torch.Size([1, 19, 1024, 2048])
                if not self.args.unscaled and (pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]):
                    pred = F.interpolate(pred, (size[-2], size[-1]), 
                                    mode='bilinear')
                if posteriors:
                    out_pred.append(pred.cpu().numpy())

                pred = pred.argmax(dim=1).cpu().numpy() # (1, 1024, 2048)
                out.append(pred)
                remaining -= cnt
                if do_tqdm: pbar.update(cnt)
        if do_tqdm: pbar.close()
        if posteriors:
            out = np.concatenate(out); out_pred = np.concatenate(out_pred)
            return out, out_pred
        print('after concatenate')
        print('end of sample')

        return np.concatenate(out), None
    
    def to_hierarchies(self, labels):
        new_labels = [labels.long()]
        for label_dict, lookup_table in zip(self.labels_dicts, self.lookup_tables):
            label = self.to_hierarchical(labels, lookup_table=lookup_table, abstain=19)
            new_labels.append(label.long())
        if torch.is_tensor(labels):
            return torch.cat(new_labels)
        else:
            return np.concatenate(new_labels)
    
    def to_hierarchical(self, labels, abstain=None, lookup_table=None):
        if lookup_table is None: # h2
            lookup_table = [0, 1, 2, 2, 2, 2, 3, 3, 2, 1, 4, 5, 5, 6, 6, 6, 6, 6, 6]+ [255]*(255-18)
        else:
            lookup_table = list(lookup_table)

            lookup_table.extend([255]*(255-18))
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

    def get_edgemap(self, label, num_classes, ignore, filer_size=5):
        label_ = label
        label_[label == ignore] = num_classes
        label_ = np.uint8(label_/label_.max()*255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filer_size, filer_size))
        edge_map = np.uint8(cv2.dilate(cv2.Canny(label_, 0, 100), kernel))
        return np.uint8(edge_map/255.0)
    
    def process_precertify(self, batch, posteriors=False):
        image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
        # get image name and label
        name = name[0].replace('_gtFine_labelIds', '')
        label = label[0]

        # normalize image
        image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)

        size = label.size()
        # image_np01 = image_np01.to(f'cuda:{self.rank}')
        baseline_pred, baseline_logits = self.sample(image_np01, size, n=1, sigma=0, posteriors=posteriors, cuda_id=self.rank) # (1, 512, 1024)
        

        flat_label = label.flatten().cpu().numpy()

        non_ignore_idx = flat_label != 255

        labels_h = self.to_hierarchies(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()

        edge_map = self.get_edgemap(copy.deepcopy(label).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL).flatten()  
        # edge_map = edge_map[non_ignore_idx]
        return name, image_np01, baseline_pred, size, labels_h, non_ignore_idx, edge_map, label, baseline_logits
    
    def fast_certify(self, samples_flattened, n0, n):
        modes, _ = torch.mode(samples_flattened[:n0], 0)
        counts_at_modes = (samples_flattened[n0:] == modes.unsqueeze(0)).sum(0)
        pvals_ = binom_test(np.array(counts_at_modes), np.array([n-n0]*len(samples_flattened[0])), np.array([self.tau]*len(samples_flattened[0])), alt='greater')
        abstain = pvals_ > self.alpha/len(samples_flattened[0])
        modes = modes.cpu().numpy()
        modes[np.array(abstain)] = 19
        return modes
    
    def get_samples(self, image_np01, n, size, name, fresh=True):
        try:
            if fresh:
                raise Exception
            #samples = pickle.load(open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_classes.pkl', 'rb'))
            logits = pickle.load(open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_logits.pkl', 'rb'))
            self.log(f'Loading cached sample {name}')
        except:
            self.log(f'Generating {n} samples for {name}')
            _, logits = self.sample(image_np01, size, n, sigma = self.sigma, posteriors=True, cuda_id=self.rank, do_tqdm=True)
            # pickle.dump(logits, open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_logits.pkl', 'wb'), protocol=4)
        return logits[:n]
    
    def info_gain(self, h_i, classes_certify, label_i, non_ignore_idx):
        c_idx = classes_certify != 19
        correct_pred = classes_certify == label_i

        filter_idx = non_ignore_idx & c_idx & correct_pred
        info_gain = (np.log(19)-np.log(self.lookup_info_gain[h_i][classes_certify[filter_idx]])).sum()

        c_ig_map = np.zeros(len(c_idx))
        c_ig_map[filter_idx] = np.log(19) - np.log(self.lookup_info_gain[0][classes_certify[filter_idx]])

        return info_gain, c_ig_map
    
    def fill_stats_h(self, stats, name, n, n0, i, classes_certify, label_i, non_ignore_idx, edge_map):
        c_idx = classes_certify != 19 
        k = (n, n0, None, i)
        stats[name][k] = {}
        c_ig, c_ig_map = self.info_gain(i, classes_certify, label_i, non_ignore_idx)
        stats[name][k]['c_info_gain'] = c_ig
        ig_per_class_dict = self.ig_per_class(c_ig_map, np.zeros(len(c_idx)), classes_certify, label_i.reshape(1, -1), non_ignore_idx)
        stats[name][k]['ig_per_class_dict'] = ig_per_class_dict

        # print(i, stats[name][(n, n0, i)]['c_info_gain']/non_ignore_idx.sum())
        stats[name][k]['num_pixels'] = non_ignore_idx.sum()
        stats[name][k]['certified_pixels'] = (classes_certify[non_ignore_idx] != 19).sum()
        stats[name][k]['certified_pos'] = (classes_certify[c_idx & non_ignore_idx] == label_i[c_idx & non_ignore_idx]).sum()

        stats[name][k]['boundary_pixels'] = (edge_map==1 & non_ignore_idx).sum()
        stats[name][k]['nonboundary_pixels'] = (edge_map==0 & non_ignore_idx).sum()

        stats[name][k]['certified_boundary_pixels'] = (edge_map==1 & c_idx & non_ignore_idx).sum()
        stats[name][k]['certified_nonboundary_pixels'] = (edge_map==0 & c_idx & non_ignore_idx).sum()

        stats[name][k]['pos_certified_boundary_pixels'] = (classes_certify[edge_map==1 & c_idx & non_ignore_idx] == label_i[edge_map==1 & c_idx & non_ignore_idx]).sum()
        stats[name][k]['pos_certified_nonboundary_pixels'] = (classes_certify[edge_map==0 & c_idx & non_ignore_idx] == label_i[edge_map==0 & c_idx & non_ignore_idx]).sum()
        return stats

    def get_difference(self, posteriors):
        n, num_classes, w, h = posteriors.shape 
        posteriors = torch.tensor(posteriors)
        posteriors_m = torch.mean(posteriors, dim=0)
        posteriors_std = torch.std(posteriors, dim=0)
        sorted_posteriors, _ = torch.sort(posteriors_m, dim=0)
        diff = sorted_posteriors[-1, :, :] - sorted_posteriors[-2, :, :]


        posteriors_std = (posteriors_std - torch.min(posteriors_std))/(torch.max(posteriors_std)-torch.min(posteriors_std))
        posteriors_m_std = torch.mean(posteriors_std, dim=0) # the average standard deviation at each pixel (w, h)

        return diff.cpu().numpy(), posteriors_m_std.cpu().numpy()
    
    
    def info_gain_adaptive(self, classes_certify, h_map, labels_h):
        non_ignore_idx = labels_h[0].flatten() != 255
        labels_h = labels_h.reshape(labels_h.shape[0], -1)
        c_idx = classes_certify != 19

        c_ig = 0
        c_ig_map = np.zeros(len(non_ignore_idx))
        for i, label_i in enumerate(labels_h):
            h_pos = (h_map == i)
            correct_pred = classes_certify == label_i
            filter_idx = non_ignore_idx & c_idx & h_pos & correct_pred
            c_ig += (np.log(19) - np.log(self.lookup_info_gain[i][classes_certify[filter_idx]])).sum()
            c_ig_map[filter_idx] = np.log(19) - np.log(self.lookup_info_gain[i][classes_certify[filter_idx]])
        return c_ig, c_ig_map
    
    def hierarchical_certified_accuracy(self, classes_certify, hierarchy, labels_h, edge_map):

        non_ignore_idx = labels_h[0] != 255

        classes_certify = classes_certify[non_ignore_idx]
        hierarchy = hierarchy[non_ignore_idx]
        labels_h = labels_h[:, non_ignore_idx]
        edge_map = edge_map[non_ignore_idx]

        num_boundary_pixels = (edge_map == 1).sum()
        num_nonboundary_pixels = (edge_map == 0).sum()

        num_pixels = non_ignore_idx.sum()

        certified_idx = classes_certify != 19

        num_certified_pixels = certified_idx.sum()
        
        classes_certify = classes_certify[certified_idx]
        hierarchy = hierarchy[certified_idx]
        labels_h = labels_h[:, certified_idx] 
        edge_map = edge_map[certified_idx]
    
        num_c_boundary_pixels = (edge_map == 1).sum()
        num_c_nonboundary_pixels = (edge_map == 0).sum()

        pos = 0
        h_acc = {}
        for i, label in enumerate(labels_h):
            h_idx = hierarchy == i
            edge_map_h = edge_map[h_idx]
            num_boundary, num_nonboundary = (edge_map_h==1).sum(), (edge_map_h==0).sum()
            num_h_pixels = h_idx.sum()
            label = label[h_idx]
            classes_c = classes_certify[h_idx]
            pos_idx = label == classes_c

            edge_map_h = edge_map_h[pos_idx]

            num_pos_c_boundary_pixels = (edge_map_h == 1).sum()
            num_pos_c_nonboundary_pixels = (edge_map_h == 0).sum()

            pos_i = (pos_idx).sum()
            h_acc[i] = {'pos': pos_i, 'num_h_pixels': num_h_pixels, 
                        'pos_certified_boundary_pixels':num_pos_c_boundary_pixels, 'pos_certified_nonboundary_pixels': num_pos_c_nonboundary_pixels,
                        'num_boundary':num_boundary, 'num_nonboundary': num_nonboundary}
            pos += pos_i
        stats = {'certified_pixels':num_certified_pixels, 'num_pixels':num_pixels, 'certified_pos':pos,
                'hierarchical_acc':h_acc,
                'boundary_pixels':num_boundary_pixels, 'nonboundary_pixels':num_nonboundary_pixels,
                'num_c_boundary_pixels':num_c_boundary_pixels, 'num_c_nonboundary_pixels':num_c_nonboundary_pixels}
        return stats
    
    def adapt_to_hierarchies(self, samples_flattened, diff, th_func=None, f=None, n=None):
        diff = diff.flatten()
        hierarchies = np.zeros_like(diff)

        if th_func is None:
            hierarchies[diff < 0.4] = 1
            hierarchies[diff < 0.3] = 2
            hierarchies[diff < 0.1] = 3
        else:
            hierarchies[diff < f[2]] = 1
            hierarchies[diff < f[1]] = 2
            hierarchies[diff < f[0]] = 3
        for i in range(len(self.lookup_tables)):
            samples_flattened[:, hierarchies == i+1] = self.lookup_tables[i][samples_flattened[:, hierarchies == i+1]]
        return samples_flattened, hierarchies
    
    def ig_per_class(self, ig_map, h_map, classes_certify, labels_h, non_ignore_idx):
        labels_h = labels_h.reshape(labels_h.shape[0], -1)
        c_idx = classes_certify != 19

        ig_per_class_dict = {} # 
        for i, label_i in enumerate(labels_h):
            h_pos = (h_map == i)
            correct_pred = classes_certify == label_i
            filter_idx = non_ignore_idx & c_idx & h_pos & correct_pred
            ig_per_class_dict[i] = {'ig': np.zeros(19), 'pixels_count': np.zeros(19)}
            labels_in_h = np.unique(label_i[non_ignore_idx])
            for j in labels_in_h:
                ig_per_class_dict[i]['ig'][j] += ig_map[filter_idx & (label_i == j)].sum()
                ig_per_class_dict[i]['pixels_count'][j] += (h_pos & non_ignore_idx & (label_i == j)).sum()
        return ig_per_class_dict
    
    def T_scaling(self, logits, args):
        temperature = args.get('temperature', None)
        return torch.div(logits, temperature)
    
    def test(self, calibration=False, temperature=1.14):
        preds = []
        labels_oneh = []
        self.steps = 0
        self.model.eval()
        for batch in tqdm(self.testloader, desc='Main Loop'):
            if self.steps < self.args.N0: self.steps +=1; continue
            if self.steps >= self.args.N: break

            image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
            # get image name and label
            name = name[0].replace('_gtFine_labelIds', '')
            label = label[0]
            # normalize image
            image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
            size = label.size()
            baseline_pred, baseline_logits = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
            flat_label = label.flatten().cpu().numpy()
            non_ignore_idx = flat_label != 255

            label_oneh = torch.nn.functional.one_hot(torch.tensor(label.cpu().numpy().flatten()[non_ignore_idx]).long(), num_classes=19)
            logits = torch.tensor(baseline_logits.reshape(19, -1)).transpose(1, 0)
            if calibration:
                logits = torch.div(logits, temperature)
            
            pred = F.softmax(logits, dim=1).cpu().numpy()[non_ignore_idx]

            labels_oneh.append(label_oneh.cpu().numpy())
            preds.append(pred)
            self.steps +=1

            if self.steps == 50: break
        preds = np.concatenate(preds).flatten()
        labels_oneh = np.concatenate(labels_oneh).flatten().flatten()

        return preds, labels_oneh
    
    def calc_bins(self, preds, labels_oneh):
        # Assign each prediction to a bin
        num_bins = 10
        bins = np.linspace(0.1, 1, num_bins)
        binned = np.digitize(preds, bins)

        # Save the accuracy, confidence and size of each bin
        bin_accs = np.zeros(num_bins)
        bin_confs = np.zeros(num_bins)
        bin_sizes = np.zeros(num_bins)

        for bin_ in range(num_bins):
            bin_sizes[bin_] = (binned == bin_).sum()
            if bin_sizes[bin_] > 0:
                bin_accs[bin_] = (labels_oneh[binned==bin_]).sum() / bin_sizes[bin_]
                bin_confs[bin_] = (preds[binned==bin_]).sum() / bin_sizes[bin_]
                print(bin_accs[bin_], bin_confs[bin_])
        return bins, binned, bin_accs, bin_confs, bin_sizes
    
    def draw_reliability_graph(self, preds, labels_oh, path):
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        ECE, MCE = self.get_metrics(preds, labels_oh)
        bins, _, bin_accs, _, _ = self.calc_bins(preds, labels_oh)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')

        # Error bars
        plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

        # Draw bars and identity line
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
        MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        plt.legend(handles=[ECE_patch, MCE_patch])

        #plt.show()
        
        plt.savefig(path, bbox_inches='tight')
    
    def get_metrics(self, preds, labels_oh):
        ECE = 0
        MCE = 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calc_bins(preds, labels_oh)

        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)

        return ECE, MCE
    
    def calibrate(self, load=False):
        if os.path.exists('calibration/calibration_temp_cs.pkl') and load: 
            T = pickle.load(open('calibration/calibration_temp_cs.pkl', 'rb'))['temperature']
            return T
        self.steps = 0
        self.model.eval()

        preds = []
        labels_oneh = []
        temperature = nn.Parameter(torch.ones(1, requires_grad=True).to(self.cdevice))
        args_ = {'temperature': temperature}
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=1000000)#, line_search_fn='strong_wolfe')

        logits_list = []
        labels_list = []
        temps = []
        losses = []

        for batch in tqdm(self.testloader, desc='Main Loop'):
            if self.steps < self.args.N0: self.steps +=1; continue
            if self.steps >= self.args.N: break

            image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
            # get image name and label
            name = name[0].replace('_gtFine_labelIds', '')
            label = label[0]

            # normalize image
            image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)

            size = label.size()

            baseline_pred, baseline_logits = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)

            flat_label = label.flatten().cpu().numpy()

            non_ignore_idx = flat_label != 255

            # label_oneh = torch.nn.functional.one_hot(torch.tensor(label.cpu().numpy().flatten()[non_ignore_idx]).long(), num_classes=19)

            logits_list.append(torch.tensor(baseline_logits.reshape(19, -1).transpose(1, 0)[non_ignore_idx]))
            labels_list.append(torch.tensor(label.flatten().cpu().numpy()[non_ignore_idx]))

            self.steps +=1
            if self.steps == 50: break

        logits_list = torch.cat(logits_list).to(self.cdevice)
        labels_list = torch.cat(labels_list).to(self.cdevice)

        def _eval():
            loss = criterion(self.T_scaling(logits_list, args_), labels_list.long())
            loss.backward()
            temps.append(temperature.item())
            losses.append(loss)
            return loss
        optimizer.step(_eval)
        print('Final T_scaling factor: {:.2f}'.format(temperature.item()))
        plt.subplot(121)
        plt.plot(list(range(len(temps))), temps)
        losses = [l.detach().cpu() for l in losses]
        plt.subplot(122)
        plt.plot(list(range(len(losses))), losses)
        os.makedirs('calibration/', exist_ok=True)
        plt.savefig('calibration/calibration.png')
        pickle.dump({'temperature': temperature.item()}, open('calibration/calibration_temp_cs.pkl', 'wb'))
        return temperature.item()
    
    def test_calibration(self):
        temperature = self.calibrate(load=False)
        preds_original, labels_oh = self.test()

        preds_calibrated, labels_oh_calib = self.test(calibration=True, temperature=temperature)

        self.draw_reliability_graph(preds_original, labels_oh, 'calibration/uncalibrated.png')
        self.draw_reliability_graph(preds_calibrated, labels_oh_calib, 'calibration/calibrated.png')

    def main_skeleton(self):
        self.steps = 0
        self.model.eval()

        self.temperature = self.calibrate(load=True)
        print(f'Using temprature = {self.temperature}')

        stats = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break

                name, image_np01, baseline_pred, size, labels_h, non_ignore_idx, edge_map, label, baseline_logits = self.process_precertify(batch, posteriors=False)
                baseline_pred = baseline_pred[0]

                samples, samples_logits = None, None
                print(image_np01.shape, labels_h.shape)
                stats[name] = {}

                #for n in list(reversed(sorted(self.n))):
                #  
                for n in tqdm(list(reversed(sorted([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]))), desc=f'{name}'):
                    if samples_logits is None:
                        print('before get_samples')
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1)#.copy()

                        samples_logits = samples_logits/self.temperature # for calibration
                        print('after get_samples')

                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]

                    samples_flattened = samples.reshape(n, -1).copy() # (N, w*h)
                    samples_flattened_h = self.to_hierarchies(torch.tensor(samples_flattened).clone().unsqueeze(0).detach())

                    # run experiment per n
                    stats_dir, stats = self.exp_best_threshold_function(name, stats,
                                      samples_flattened,
                                      samples_flattened_h, labels_h, 
                                      non_ignore_idx, edge_map,
                                      samples_logits,
                                      label, baseline_pred)
                pickle.dump(stats, open(os.path.join(stats_dir, f'{name}.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Saving {os.path.join(stats_dir, f"{name}.pkl")}')
                self.steps +=1

    def exp_best_threshold_function(self, name, stats,
                                      samples_flattened,
                                      samples_flattened_h, labels_h, 
                                      non_ignore_idx, edge_map,
                                      samples_logits,
                                      label, baseline_pred):
        stats_dir = f'{self.default_log_dir}/slurm_logs/exp_best_threshold_function/'
        os.makedirs(stats_dir, exist_ok=True)
        n, _ = samples_flattened.shape


        n0=20; N0=n0
        i = 0
        # SegCertify at different hierarchies
        for s_i, label_i in tqdm(zip(samples_flattened_h, labels_h), desc='SegCertify'):
            classes_certify = self.fast_certify(s_i, n0, n)
            
            label_i = label_i.flatten()
            stats = self.fill_stats_h(stats, name, n, N0, i, classes_certify, label_i, non_ignore_idx, edge_map)
            i+=1
        # i = 4 --> adaptive
        posteriors = F.softmax(torch.tensor(samples_logits[:n0]), dim=1).cpu().numpy()
        diff, _ = self.get_difference(posteriors[:n0])
        from itertools import permutations
        def get_threshold_permutations(l):
            x_sorted = []
            x = list(permutations(l, 3))
            for f in x:
                if np.all(np.diff(f) >= 0):
                    x_sorted.append(f)
            return x_sorted
        th_functions = get_threshold_permutations([0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
        print(f'Using {len(th_functions)} threshold functions.')
        for f in tqdm(th_functions, desc=f'{name} threshold functions:'):
            th_func_name = str(f)
            samples_flattened_adaptive, h_map = self.adapt_to_hierarchies(samples_flattened.copy(), diff, th_func_name, f, n)
            classes_certify = self.fast_certify(torch.tensor(samples_flattened_adaptive), n0, n).copy()
            stats_h = self.hierarchical_certified_accuracy(classes_certify.copy(), h_map, labels_h.reshape(labels_h.shape[0], -1), self.get_edgemap(copy.deepcopy(label).cpu().numpy(), 20, 255).flatten())
            ig_adaptive, ig_map = self.info_gain_adaptive(classes_certify, h_map, labels_h)
            # get adaptive baseline
            ig_per_class_dict = self.ig_per_class(ig_map, h_map, classes_certify, labels_h, non_ignore_idx) # verified
            label__ = np.reshape(np.array(label).flatten()[non_ignore_idx], (1, -1))
            baseline_pred__ = np.reshape(np.array(baseline_pred).flatten()[non_ignore_idx], (1, -1))
            label_adaptive, h_map = self.adapt_to_hierarchies(label__, diff.flatten()[non_ignore_idx])
            baseline_pred_adaptive, h_map = self.adapt_to_hierarchies(baseline_pred__, diff.flatten()[non_ignore_idx])

            k = (n, N0, th_func_name, 4)
            stats[name][k] = stats_h
            stats[name][k]['ig_per_class_dict'] = ig_per_class_dict
            stats[name][k]['baseline_pos'] = (label_adaptive == baseline_pred_adaptive).sum()
            stats[name][k]['c_info_gain'] = ig_adaptive
            # print(i, info_gain_adaptive/non_ignore_idx.sum())
            # print(th_func, stats[name][(n, N0, None, 0)]['c_info_gain']/non_ignore_idx.sum(), ig_adaptive/non_ignore_idx.sum())
        return stats_dir, stats
    def exp_adaptive_n0_and_thresholds(self, name, stats,
                                      samples_flattened,
                                      samples_flattened_h, labels_h, 
                                      non_ignore_idx, edge_map,
                                      samples_logits,
                                      label, baseline_pred):
        stats_name = 'slurm_adaptive_n0_and_thresholds'
        os.makedirs(f'{self.default_log_dir}/slurm_logs/{stats_name}/', exist_ok=True)
        n, _ = samples_flattened.shape

        for N0 in [10, 20, 0.05, 0.1, 0.2]:
            if N0 < 1: n0 = int(N0*n)
            else: n0 = N0
        
            i = 0
            for s_i, label_i in zip(samples_flattened_h, labels_h):
                classes_certify = self.fast_certify(s_i, n0, n)
                
                label_i = label_i.flatten()
                stats = self.fill_stats_h(stats, name, n, N0, i, classes_certify, label_i, non_ignore_idx, edge_map)
                i+=1
            # i = 4 --> adaptive
            posteriors = F.softmax(torch.tensor(samples_logits[:n0]), dim=1).cpu().numpy()
            diff, _ = self.get_difference(posteriors[:n0])
            for th_func in self.threshold_functions.keys():
                samples_flattened_adaptive, h_map = self.adapt_to_hierarchies(samples_flattened.copy(), diff, th_func, n)
                classes_certify = self.fast_certify(torch.tensor(samples_flattened_adaptive), n0, n).copy()
                stats_h = self.hierarchical_certified_accuracy(classes_certify.copy(), h_map, labels_h.reshape(labels_h.shape[0], -1), self.get_edgemap(copy.deepcopy(label).cpu().numpy(), 20, 255).flatten())
                ig_adaptive, ig_map = self.info_gain_adaptive(classes_certify, h_map, labels_h)
                # get adaptive baseline
                ig_per_class_dict = self.ig_per_class(ig_map, h_map, classes_certify, labels_h, non_ignore_idx) # verified
                label__ = np.reshape(np.array(label).flatten()[non_ignore_idx], (1, -1))
                baseline_pred__ = np.reshape(np.array(baseline_pred).flatten()[non_ignore_idx], (1, -1))
                label_adaptive, h_map = self.adapt_to_hierarchies(label__, diff.flatten()[non_ignore_idx])
                baseline_pred_adaptive, h_map = self.adapt_to_hierarchies(baseline_pred__, diff.flatten()[non_ignore_idx])

                k = (n, N0, th_func, 4)
                stats[name][k] = stats_h
                stats[name][k]['ig_per_class_dict'] = ig_per_class_dict
                stats[name][k]['baseline_pos'] = (label_adaptive == baseline_pred_adaptive).sum()
                stats[name][k]['c_info_gain'] = ig_adaptive
                # print(i, info_gain_adaptive/non_ignore_idx.sum())
                # print(th_func, stats[name][(n, N0, None, 0)]['c_info_gain']/non_ignore_idx.sum(), ig_adaptive/non_ignore_idx.sum())
        return stats_name, stats
    
    def ig_per_gt_class(self, classes_certify, info_gain_map, hierarchy_map, labels_h):
        non_ignore_idx = labels_h[0].flatten() != 255
        notc_idx = classes_certify == 19
        c_idx = classes_certify != 19
        labels_h = labels_h.reshape(labels_h.shape[0], -1)

        ig_per_h_ls = []
        for i, label in enumerate(labels_h):
            h_idx = hierarchy_map == i
            ig_per_class = np.zeros(19)
            count_per_class = np.zeros(19)
            cur_map = info_gain_map[h_idx]
            for c_i in range(19):
                class_pos = (label[h_idx].copy() == c_i)
                class_ig_enteries = cur_map[(classes_certify[h_idx] == label[h_idx]) & class_pos]
                ig_per_class[c_i] = class_ig_enteries.sum()
                count_per_class[c_i] = (label[h_idx & non_ignore_idx].copy() == c_i).sum()

            ig_per_h_ls.append({'ig_per_class':ig_per_class, 'count_gt_pixels':count_per_class})

        return ig_per_h_ls
    

    def log(self, msg):
        print(f'GPU{self.rank}, job {self.jobid}: {msg}')
    
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    remote = False
    parser = ArgumentParser()
    parser.add_argument("--jobid", default=1)
    parser.add_argument("--numjobs", default=1)

    args, _ = parser.parse_known_args()
    print(args.jobid)
    del parser
    if remote:
        torch.multiprocessing.spawn(
            Certifier,
            nprocs=world_size,
            args=(world_size, args.jobid, args.numjobs),)
    else:
        trainer = Certifier(0, 1, args.jobid, args.numjobs)