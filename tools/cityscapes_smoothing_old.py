
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
    def __init__(self, rank, world_size):
        self.rank, self.world_size = rank, world_size
        self.parse_args()
        self.init_model()
        self.init_datasets()
        self.init_writer()
        self.init_pre_certify()
        #self.fast_certify()
        self.paper_adaptive_n0()
        # self.paper_logs()
        # self.baseline_hs()
        
        
        #self.run_baseline()
        #self.aggregated_relaxed_certify()
        #self.aggregated()
        #self.test_certification_jump()
        #self.gen_cached_samples()
        #self.dataset_certify_h()
        #self.dataset_validate()

    def parse_args(self):
        parser = ArgumentParser()
        parser = Certifier.add_argparse_args(parser)
        parser = ConfigArgumentParser(parents=[parser], add_help=False)
        parser.add_argument("--config", default='/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/configs/cityscapes.ini', is_config_file=True)
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
        parser.add_argument('--default_log_dir', type=str, default= '/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes1024')
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--smooth_kernel_size', default=7)
        parser.add_argument('--hierarchical', action="store_true")
        parser.add_argument('--relaxation', action="store_true")
        parser.add_argument('--smoothing', action="store_true")
        parser.add_argument('--window_size', type=int, default=5)
        parser.add_argument('--cfg', type=str, default='/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/configs/cityscapes.yml')
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
        self.labels_dict = pickle.load(open('/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/data/cityscapes_trainIds.pkl', 'rb'))
        self.labels_dict_h = {0:'drivable', 1:'flat non-drivable', 2:'obstacle', 3:'road sign', 4:'sky', 5:'human', 6:'vehicle', 7:'abstain'}
        self.labels_dict = pickle.load(open('/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/data/cityscapes_trainIds.pkl', 'rb'))
        self.labels_dict[19] = 'abstain'

        self.labels_dict_h1 = self.labels_dict # 19
        self.labels_dict_h2 = {0:'drivable', 1:'sky', 2:'construction & vegetation', 3:'traffic sign', 4:'human', 5:'vehicle', 6:'flat obstacle', 7:'abstain'} # 7
        self.labels_dict_h3 = {0: 'drivable', 1:'static obstacle', 2: 'dynamic obstacle', 3:'flat obstacle', 4: 'abstain'} # 5
        self.labels_dict_h4 = {0: 'drivable', 1:'obstacle', 2:'abstain'} # 3
        self.lookup_tables = [np.array([0, 6, 2, 2, 2, 2, 3, 3, 2, 6, 1, 4, 4, 5, 5, 5, 5, 5, 5]),
                              np.array([0, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 2, 2]),
                              np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])]
        self.lookup_info_gain = [np.array([0, 0, 5, 2, 2, 6, 2]),
                                np.array([0, 8, 8, 2]),
                                np.array([0, 18])]
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
        experiment_folder = f'{self.default_log_dir}/version_{len(os.listdir(self.default_log_dir))+100}'
        os.makedirs(experiment_folder, exist_ok=True)
        print(experiment_folder)
        shutil.copyfile(self.args.config, f'{experiment_folder}/config.ini')
        self.writer = SummaryWriter(experiment_folder)

    def load_model(self, config):
        # build model
        model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)
        model.to(self.cdevice)
        model_state_file = config.TEST.MODEL_FILE 
        pretrained_dict = torch.load(model_state_file)#, map_location=self.cdevice)
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = nn.DataParallel(model, device_ids=self.gpus).cuda()
        return model
    
    def load_data(self, config, args):
        test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0]) # (1024, 2048)
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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
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
    
    def run_baseline(self):
        self.steps = 0
        agg_stats = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                label = label[0]
                size = label.size()
                baseline_pred, _ = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
                baseline_pred = baseline_pred[0]
                confusion_matrix, baseline_acc = self.get_confusion_matrix(label, baseline_pred, self.num_classes, self.config.TRAIN.IGNORE_LABEL)

                agg_stats[name] = {'acc':baseline_acc, 'confusion_matrix': confusion_matrix}
        
        pickle.dump(agg_stats, open(f'{self.default_log_dir}/agg_stats/baseline.pkl', 'wb'))
    
    def run_certify(self, sample, n, tau, parallel=True, correction='bonferroni', kfwer_k=1, greedy=True, max_size=1):
        assert isinstance(sample, np.ndarray)
        shape = sample.shape # (1100, 2097152)
        assert len(shape) == 2
        t0 = time.time()
        K = shape[1] # nr points (number of pixels = 1024*2048)
        if parallel:
            if greedy:
                with mp.Pool() as pool:
                    c_ps = pool.map(get_mode_and_count_with_ignore, [(sample[:, i], n, tau, self.alpha/sample.shape[1], max_size, self.num_classes) for i in range(K)]) # ((1100,), 100, 1000, 19, 0.75)
            else:
                with mp.Pool() as pool:
                    c_ps = pool.map(map_test, [(sample[:, i], n, tau) for i in range(K)]) # ((1100,), 100, 1000, 19, 0.75)
        else:
            c_ps = map(self.map_test, [(sample[:, i], n, tau) for i in range(K) ])

        # len(c_ps)=2097152, made of 2-tuples (mode-class, binom_test result)
        classes, ps, count_map = zip(*c_ps)
        classes = np.array(list(classes))
        ps = np.array(list(ps))
        if not greedy:
            if correction == 'kfwer':
                sortind = np.argsort(ps)
                pvals = np.take(ps, sortind)
                ntests = len(pvals)
                alpha_ = kfwer_k * self.alpha / np.arange(ntests, 0, -1)
                alpha_[:kfwer_k] = kfwer_k * self.alpha / ntests
                notreject = pvals > alpha_
                nr_index = np.nonzero(notreject)[0]
                if nr_index.size == 0:
                    # nonreject is empty, all rejected
                    notrejectmin = len(pvals)
                else:
                    notrejectmin = np.min(nr_index)
                notreject[notrejectmin:] = True
                reject_ = ~notreject
                reject = np.empty_like(reject_)
                reject[sortind] = reject_
            else:
                reject, _, _ , _= multipletests(ps, alpha=self.alpha, method=correction)
            I = np.logical_not(reject)
            classes[I] = self.abstain_label

        radius = self.sigma * sps.norm.ppf(tau)
        return classes, ps, radius, (0, 0), count_map
    
    def run_relaxed_certify(self, sample, n0, n, tau, parallel):
        assert isinstance(sample, np.ndarray)
        shape = sample.shape # (1100, 2097152)
        assert len(shape) == 2

        K = shape[1] # nr points (number of pixels = 1024*2048)
        if parallel:
            with mp.Pool() as pool:
                c_ps = pool.map(pixel_test, [(sample[:, i], n0, n, tau, self.alpha/sample.shape[1], self.lookup_tables) for i in range(K)]) # ((1100,), 100, 1000, 19, 0.75)
        else:
            c_ps = map(pixel_test, [(sample[:, i], n0, n, tau, self.alpha/sample.shape[1], self.lookup_tables) for i in range(K)])

        # len(c_ps)=2097152, made of 2-tuples (mode-class, binom_test result)
        classes, hierarchy,  = zip(*c_ps)
        classes = np.array(list(classes))
        hierarchy = np.array(list(hierarchy))
        radius = self.sigma * sps.norm.ppf(tau)
        return classes, hierarchy, radius
    
    def certified_accuracy(self, oh, label):
        l = torch.tensor(label.reshape(-1))
        oh = torch.tensor(oh)
        ignore_idx = l != self.config.TRAIN.IGNORE_LABEL
        oh = oh[ignore_idx]
        ig_idx = oh.argmax(1) != self.abstain_label
        num_pixels = oh.shape[0]
        oh = oh[ig_idx]
        l = l[ignore_idx]
        l = l[ig_idx]

        cs = oh[np.arange(oh.shape[0]), l.long()]
        num_certified_pixels = cs.shape[0]
        return (sum(cs)/num_pixels).item(), (sum(cs)/num_certified_pixels).item(), sum(cs)
    
        
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

    def aggregated_relaxed_certify(self):
        self.steps = 0
        #self.model.eval()

        baseline_stats = {}
        agg_stats = {}
        baseline_only = False
        abstain_label = 19
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gt_labelIds', '')

                size = label.size()
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                edge_map = self.get_edgemap(copy.deepcopy(label[0]).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL)    
                labels_h = self.to_hierarchies(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()

                #orig_label = label.cpu().numpy()
                #flat_label = label.flatten().cpu().numpy()
                #baseline_pred, _ = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
                #confusion_matrix, baseline_acc = self.get_confusion_matrix(label[0], baseline_pred[0], self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                #agg_stats[name] = {'acc':baseline_acc, 'confusion_matrix': confusion_matrix}

                #baseline_preds_h = self.to_hierarchies(torch.tensor(baseline_pred).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()

                # i=0
                # baseline_stats[name] = {}
                # for label_i, baseline_pred_i in zip(labels_h, baseline_preds_h):
                #     if i==0:
                #         num_classes = 20
                #     else:
                #         num_classes = len(self.labels_dicts[i-1].keys())
                #     confusion_matrix, baseline_acc = self.get_confusion_matrix(label_i, baseline_pred_i, self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                    
                #     baseli
                if baseline_only:
                    continue
                samples_logits, samples = None, None
                for n in tqdm(self.n, desc='ns'):
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1)
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]
                    samples_flattened = samples.reshape(samples.shape[0], -1) # (N, w*h)
                    classes_certify, hierarchy, radius = self.run_relaxed_certify(samples_flattened, 10, n, self.tau, parallel=True) # (w*h, 20)
                    stats = self.hierarchical_certified_accuracy(classes_certify, hierarchy, labels_h.reshape(labels_h.shape[0], -1), edge_map.flatten()) 
                    if name not in agg_stats:
                        agg_stats[name] = {}
                    agg_stats[name][n] = stats
                pickle.dump(agg_stats, open(f'{self.default_log_dir}/agg_stats/stats_h_{self.steps}.pkl', 'wb'))
                self.steps +=1

        #pickle.dump(baseline_stats, open(f'{self.default_log_dir}/agg_stats/baseline_h.pkl', 'wb'))
        #pickle.dump(agg_stats, open(f'{self.default_log_dir}/agg_stats/baseline.pkl', 'wb'))
    
    def diff_vs_setsize(self):
        self.steps = 0
        #self.model.eval()
        os.makedirs(f'{self.default_log_dir}/agg_stats/', exist_ok=True)

        n0 = 20
        diff_stats = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                size = label.size()
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                samples, samples_logits = None, None
                n = 300
                if samples_logits is None:
                    samples_logits = self.get_samples(image_np01, n, size, name)
                    samples = samples_logits.argmax(1)
                else:
                    samples, samples_logits = samples[0:n], samples_logits[0:n]
                samples_flattened = samples.reshape(samples.shape[0], -1) # (N, w*h)
                classes_certify_oh, pvals, radius, timings, cnt = self.run_certify(samples_flattened, n, self.tau, correction='bonferroni', greedy=self.greedy, max_size=19) # (w*h, 20)
                set_sizes = classes_certify_oh.sum(1)
                posteriors = F.softmax(torch.tensor(samples_logits[:n0]), dim=1).cpu().numpy()
                diff, posteriors_m_std = self.get_difference(posteriors[:n0])
                diff = diff.flatten()
                diff_stats[name] = (set_sizes, diff, posteriors_m_std)
                pickle.dump(diff_stats, open(f'{self.default_log_dir}/agg_stats/diff_stats_{self.steps}', 'wb'))
                self.steps +=1
    
    def fast_certify(self, samples_flattened, n0, n):
        modes, _ = torch.mode(samples_flattened[:n0], 0)
        counts_at_modes = (samples_flattened[n0:] == modes.unsqueeze(0)).sum(0)
        pvals_ = binom_test(np.array(counts_at_modes), np.array([n-n0]*len(samples_flattened[0])), np.array([self.tau]*len(samples_flattened[0])), alt='greater')
        abstain = pvals_ > self.alpha/len(samples_flattened[0])
        modes = modes.cpu().numpy()
        modes[np.array(abstain)] = 19
        return modes
    
    def adapt_to_hierarchies(self, samples_flattened, diff, th_func=None, n=None):
        diff = diff.flatten()
        hierarchies = np.zeros_like(diff)

        if th_func is None:
            hierarchies[diff < 0.4] = 1
            hierarchies[diff < 0.3] = 2
            hierarchies[diff < 0.1] = 3
        else:
            if '_n' in th_func:
                f = self.threshold_functions[th_func](n)
            else:
                f = self.threshold_functions[th_func]
            hierarchies[diff < f[2]] = 1
            hierarchies[diff < f[1]] = 2
            hierarchies[diff < f[0]] = 3
        for i in range(len(self.lookup_tables)):
            samples_flattened[:, hierarchies == i+1] = self.lookup_tables[i][samples_flattened[:, hierarchies == i+1]]
        return samples_flattened, hierarchies
    
    def baseline_hs(self):
        # paper logs
        self.steps = 0
        self.model.eval()
        n0 = 20
        stats = {}

        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                label = label[0]
                size = label.size()
                baseline_pred, _ = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
                baseline_pred = baseline_pred[0]
                labels_h = self.to_hierarchies(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()
                baseline_pred_h = self.to_hierarchies(torch.tensor(baseline_pred).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()
                if name not in stats:
                    stats[name] = {}
                h = 0
                for label, baseline_pred in zip(labels_h, baseline_pred_h):
                    confusion_matrix, baseline_acc = self.get_confusion_matrix(label, baseline_pred, self.num_classes, self.config.TRAIN.IGNORE_LABEL)  
                    info_gain = self.info_gain(h, baseline_pred.flatten(), label.flatten())/(label !=255).sum()
                    stats[name][h] = {'acc': baseline_acc, 'confusion_matrix': confusion_matrix, 'info_gain': info_gain}
                    h +=1
                self.steps += 1
        pickle.dump(stats, open(f'{self.default_log_dir}/paper_logs/baseline_hs_{self.steps}.pkl', 'wb'))
    def info_gain(self, h_i, classes_certify, label_i):
        c_idx = classes_certify != 19
        if h_i == 0: 
            info_gain = (classes_certify == label_i).sum()*np.log(19)
        else:
            classes_certify_certified = classes_certify[c_idx].copy()
            tmp = self.lookup_info_gain[h_i - 1][classes_certify_certified]
            tmp[tmp==0] = 1
            info_gain = (np.log(19)-np.log(tmp[classes_certify[c_idx] == label_i[c_idx]])).sum()

        return info_gain
    
    def info_gain_adaptive(self, classes_certify, hierarchy_map, labels_h):
        non_ignore_idx = labels_h[0].flatten() != 255
        num_pixels = non_ignore_idx.sum()
        classes_certify = classes_certify[non_ignore_idx]
        hierarchy_map = hierarchy_map[non_ignore_idx]
        labels_h = labels_h.reshape(labels_h.shape[0], -1)[:, non_ignore_idx]

        c_idx = classes_certify != 19
        info_gain = 0
        for i, label in enumerate(labels_h):
            h_idx = hierarchy_map == i
            info_gain += self.info_gain(i, classes_certify[h_idx & c_idx].copy(), label.flatten()[h_idx & c_idx].copy())
            print(i, self.info_gain(i, classes_certify[h_idx & c_idx].copy(), label.flatten()[h_idx & c_idx].copy()))
        return info_gain
    

    def info_gain_map(self, classes_certify, hierarchy_map, labels_h):
        non_ignore_idx = labels_h[0].flatten() != 255
        ig_map = np.zeros(512*1024)
        notc_idx = classes_certify == 19
        c_idx = classes_certify != 19
        labels_h = labels_h.reshape(labels_h.shape[0], -1)

        for i, label in enumerate(labels_h):
            h_idx = hierarchy_map == i
            ig_map[h_idx & non_ignore_idx & c_idx] = self.info_gain_h(i, classes_certify[h_idx & non_ignore_idx & c_idx].copy(), 
                                                        label.flatten()[h_idx & non_ignore_idx & c_idx].copy())
            print('map', i, ig_map[h_idx & non_ignore_idx & c_idx].sum())

        return ig_map

    def info_gain_h(self, h_i, classes_certify, label_i):
        ig_map = np.zeros_like(classes_certify)
        if h_i == 0: 
            ig_map = (classes_certify == label_i)*np.log(19)
        else:
            tmp = self.lookup_info_gain[h_i - 1][classes_certify]
            tmp[tmp==0] = 1
            ig_map[classes_certify== label_i] = np.log(19)-np.log(tmp[classes_certify== label_i])
        return ig_map
    
    def adaptive_seg_palette(self, classes_certify, h_map):
        import colorsys
        import random

        def _gen_random_colors(N, bright=False):
            brightness = 1.0 if bright else 0.5
            hsv = [(i / N, 1, brightness) for i in range(N)]
            colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
            random.shuffle(colors)
            return np.array(colors)
        import seaborn as sns


        ps = [np.array(sns.color_palette()), np.array(sns.color_palette("Set2")), np.array(sns.color_palette("pastel"))]
        h_map = h_map.reshape(512, 1024)
        h_map[classes_certify.reshape(512, 1024)==19] = 255

        classes_certify = classes_certify.reshape(512, 1024)
        seg = self.get_seg_palette(classes_certify)[0].transpose(1, 2, 0)
        for i in range(1, 4):
            seg[h_map == i] = ps[i-1][classes_certify[h_map==i]]*255
        seg[classes_certify == 19] = 255
        # h1 = (170, 192, 240)
        # h2 = (148, 179, 154)
        # h3 = (179, 159, 207)
        # print(seg.shape)
        # seg[h_map==1] = h1
        # seg[h_map==2] = h2
        # seg[h_map == 3] = h3
        return seg
    def h_map_palette(self, h_map):
        h_map = h_map.reshape(512, 1024)
        import seaborn as sns
        p = np.array(sns.color_palette("flare"))

        return p[h_map.astype(np.uint8)]*255
    
    def abstain_map(self, classes_certify):
        classes_certify = classes_certify.reshape(512, 1024)
        x = np.zeros((512, 1024))
        x[classes_certify == 19] = 1
        return x*255
    
    def process_precertify(self, batch):
        image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
        name = name[0].replace('_gtFine_labelIds', '')
        # if name != 'frankfurt_000000_005543': continue
        image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
        label = label[0]
        size = label.size()
        baseline_pred, _ = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
        baseline_pred = baseline_pred[0]
        flat_label = label.flatten().cpu().numpy()
        non_ignore_idx = flat_label != 255
        labels_h = self.to_hierarchies(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()
        edge_map = self.get_edgemap(copy.deepcopy(label).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL).flatten()  
        edge_map = edge_map[non_ignore_idx]
        return name, image_np01, baseline_pred, size, labels_h, non_ignore_idx, edge_map, label
    
    def fill_stats_h(self, stats, name, n, n0, i, classes_certify, label_i, non_ignore_idx, c_idx, edge_map):
        stats[name][(n, n0, None, i)] = {}
        stats[name][(n, n0, None, i)]['c_info_gain'] = self.info_gain(i, classes_certify, label_i)
        # print(i, stats[name][(n, n0, i)]['c_info_gain']/non_ignore_idx.sum())
        stats[name][(n, n0, None, i)]['num_pixels'] = len(classes_certify)
        stats[name][(n, n0, None, i)]['certified_pixels'] = (classes_certify != 19).sum()
        stats[name][(n, n0, None, i)]['certified_pos'] = (classes_certify[c_idx] == label_i[c_idx]).sum()

        stats[name][(n, n0, None, i)]['boundary_pixels'] = (edge_map==1).sum()
        stats[name][(n, n0, None, i)]['nonboundary_pixels'] = (edge_map==0).sum()

        stats[name][(n, n0, None, i)]['certified_boundary_pixels'] = (edge_map==1 & c_idx).sum()
        stats[name][(n, n0, None, i)]['certified_nonboundary_pixels'] = (edge_map==0 & c_idx).sum()

        stats[name][(n, n0, None, i)]['pos_certified_boundary_pixels'] = (classes_certify[edge_map==1 & c_idx] == label_i[edge_map==1 & c_idx]).sum()
        stats[name][(n, n0, None, i)]['pos_certified_nonboundary_pixels'] = (classes_certify[edge_map==0 & c_idx] == label_i[edge_map==0 & c_idx]).sum()
        return stats

    def paper_adaptive_n0(self):
        self.steps = 0
        stats_name = 'adaptive_n0_th_func'
        self.model.eval()
        stats = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                name, image_np01, baseline_pred, size, labels_h, non_ignore_idx, edge_map, label = self.process_precertify(batch)

                samples, samples_logits = None, None
                stats[name] = {}

                for n in tqdm(list(reversed(sorted(self.n))), desc='ns'):
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1).copy()
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]
                    for N0 in [10, 20, 0.05, 0.1, 0.2]:
                        if N0 < 1: n0 = int(N0*n)
                        else: n0 = N0

                        
                        samples_flattened = samples.reshape(n, -1).copy() # (N, w*h)
                        samples_flattened_h = self.to_hierarchies(torch.tensor(samples_flattened).clone().unsqueeze(0).detach())
                    
                        i = 0
                        for s_i, label_i in zip(samples_flattened_h, labels_h):
                            classes_certify = self.fast_certify(s_i, n0, n)

                            classes_certify = classes_certify[non_ignore_idx]
                            c_idx = classes_certify != 19
                            label_i = label_i.flatten()[non_ignore_idx]
                            stats = self.fill_stats_h(stats, name, n, N0, i, classes_certify, label_i, non_ignore_idx, c_idx, edge_map)
                            i+=1
                        # i = 4 --> adaptive
                        posteriors = F.softmax(torch.tensor(samples_logits[:n0]), dim=1).cpu().numpy()
                        diff, _ = self.get_difference(posteriors[:n0])
                        for th_func in self.threshold_functions.keys():
                            samples_flattened_adaptive, hierarchy_map = self.adapt_to_hierarchies(samples_flattened.copy(), diff, th_func, n)
                            classes_certify = self.fast_certify(torch.tensor(samples_flattened_adaptive), n0, n).copy()
                            stats_h = self.hierarchical_certified_accuracy(classes_certify.copy(), hierarchy_map, labels_h.reshape(labels_h.shape[0], -1), self.get_edgemap(copy.deepcopy(label).cpu().numpy(), 20, 255).flatten())
                            info_gain_adaptive = self.info_gain_adaptive(classes_certify, hierarchy_map, labels_h)
                            info_gain_map = self.info_gain_map(classes_certify, hierarchy_map, labels_h)
                            ig_hist = self.ig_per_gt_class(classes_certify, info_gain_map, hierarchy_map, labels_h)
                            # get adaptive baseline
                            label__ = np.reshape(np.array(label).flatten()[non_ignore_idx], (1, -1))
                            baseline_pred__ = np.reshape(np.array(baseline_pred).flatten()[non_ignore_idx], (1, -1))
                            label_adaptive, hierarchy_map = self.adapt_to_hierarchies(label__, diff.flatten()[non_ignore_idx])
                            baseline_pred_adaptive, hierarchy_map = self.adapt_to_hierarchies(baseline_pred__, diff.flatten()[non_ignore_idx])

                            stats[name][(n, N0, th_func, 4)] = stats_h
                            stats[name][(n, N0, th_func, 4)]['baseline_pos'] = (label_adaptive == baseline_pred_adaptive).sum()
                            stats[name][(n, N0, th_func, 4)]['c_info_gain'] = info_gain_adaptive
                            # print(i, info_gain_adaptive/non_ignore_idx.sum())
                            print(stats[name][(n, N0, None, 0)]['c_info_gain']/non_ignore_idx.sum(), info_gain_adaptive/non_ignore_idx.sum())
                pickle.dump(stats, open(f'{self.default_log_dir}/paper_logs/{stats_name}_{self.steps}.pkl', 'wb'))
                self.steps +=1
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

    
    def paper_logs(self):
        self.steps = 0
        self.model.eval()
        n0 = 20
        stats = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                # if name != 'frankfurt_000000_005543': continue
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                label = label[0]
                size = label.size()
                cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_label.png', self.get_seg_palette(label)[0].transpose(1, 2, 0)[...,::-1])
                cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}.png', image_np01[...,::-1]*255)
                baseline_pred, _ = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
                baseline_pred = baseline_pred[0]
                cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_baseline.png', self.get_seg_palette(baseline_pred)[0].transpose(1, 2, 0)[...,::-1])

                flat_label = label.flatten().cpu().numpy()
                non_ignore_idx = flat_label != 255
                labels_h = self.to_hierarchies(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()
                edge_map = self.get_edgemap(copy.deepcopy(label).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL).flatten()  

                edge_map = edge_map[non_ignore_idx]
                samples, samples_logits = None, None
                stats[name] = {}
                for n in tqdm(self.n, desc='ns'):
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1).copy()
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]
                    
                    samples_flattened = samples.reshape(n, -1).copy() # (N, w*h)
                    samples_flattened_h = self.to_hierarchies(torch.tensor(samples_flattened).clone().unsqueeze(0).detach())

                    

                    i = 0
                    for s_i, label_i in zip(samples_flattened_h, labels_h):
                        classes_certify = self.fast_certify(s_i, n0, n)
                        if i == 0:
                            classes_certify_rgb_sc = self.get_seg_palette(classes_certify.reshape(512, 1024))[0].transpose(1, 2, 0).copy()
                            cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_{n}_segcertify.png', classes_certify_rgb_sc[...,::-1])
                            abstain_map_sc = self.abstain_map(classes_certify)
                            cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_{n}_segcertify_amap.png', abstain_map_sc)

                        classes_certify = classes_certify[non_ignore_idx]
                        c_idx = classes_certify != 19
                        label_i = label_i.flatten()[non_ignore_idx]
                        stats[name][(n-n0, i)] = {}
                        stats[name][(n-n0, i)]['c_info_gain'] = self.info_gain(i, classes_certify, label_i)
                        print(i, stats[name][(n-n0, i)]['c_info_gain']/non_ignore_idx.sum())
                        stats[name][(n-n0, i)]['num_pixels'] = len(classes_certify)
                        stats[name][(n-n0, i)]['certified_pixels'] = (classes_certify != 19).sum()
                        stats[name][(n-n0, i)]['certified_pos'] = (classes_certify[c_idx] == label_i[c_idx]).sum()

                        stats[name][(n-n0, i)]['boundary_pixels'] = (edge_map==1).sum()
                        stats[name][(n-n0, i)]['nonboundary_pixels'] = (edge_map==0).sum()

                        stats[name][(n-n0, i)]['certified_boundary_pixels'] = (edge_map==1 & c_idx).sum()
                        stats[name][(n-n0, i)]['certified_nonboundary_pixels'] = (edge_map==0 & c_idx).sum()

                        stats[name][(n-n0, i)]['pos_certified_boundary_pixels'] = (classes_certify[edge_map==1 & c_idx] == label_i[edge_map==1 & c_idx]).sum()
                        stats[name][(n-n0, i)]['pos_certified_nonboundary_pixels'] = (classes_certify[edge_map==0 & c_idx] == label_i[edge_map==0 & c_idx]).sum()
                        i+=1

                    # i = 4 --> adaptive
                    posteriors = F.softmax(torch.tensor(samples_logits[:n0]), dim=1).cpu().numpy()
                    diff, _ = self.get_difference(posteriors[:n0])
                    samples_flattened_adaptive, hierarchy_map = self.adapt_to_hierarchies(samples_flattened.copy(), diff)
                    classes_certify = self.fast_certify(torch.tensor(samples_flattened_adaptive), n0, n).copy()
                    abstain_map = self.abstain_map(classes_certify)
                    cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_{n}_adaptive_amap.png', abstain_map)

                    classes_certify_rgb_adaptive = self.adaptive_seg_palette(classes_certify, hierarchy_map.copy())
                    cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_{n}_adaptive.png', classes_certify_rgb_adaptive[...,::-1])
                    h_map_rgb = self.overlay_heatmap(hierarchy_map.reshape(512, 1024), image_np01[...,::-1]*255, alpha=0.7).transpose(1, 2, 0)
                    cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_{n}_hmap.png', h_map_rgb[...,::-1])
                    diff_hmap = self.overlay_heatmap(diff, image_np01[...,::-1]*255).transpose(1, 2, 0)
                    cv2.imwrite(f'/BS/mlcysec2/work/robust-segmentation/code/hrnet_seg/cityscapes_logs/hook/{name}_{n}_diff.png', diff_hmap)
                    stats_h = self.hierarchical_certified_accuracy(classes_certify, hierarchy_map, labels_h.reshape(labels_h.shape[0], -1), self.get_edgemap(copy.deepcopy(label).cpu().numpy(), 20, 255).flatten())
                    info_gain_adaptive = self.info_gain_adaptive(classes_certify, hierarchy_map, labels_h)
                    # get adaptive baseline
                    label__ = np.reshape(np.array(label).flatten()[non_ignore_idx], (1, -1))
                    baseline_pred__ = np.reshape(np.array(baseline_pred).flatten()[non_ignore_idx], (1, -1))
                    label_adaptive, hierarchy_map = self.adapt_to_hierarchies(label__, diff.flatten()[non_ignore_idx])
                    baseline_pred_adaptive, hierarchy_map = self.adapt_to_hierarchies(baseline_pred__, diff.flatten()[non_ignore_idx])

                    stats[name][(n-n0, 4)] = stats_h
                    stats[name][(n-n0, 4)]['baseline_pos'] = (label_adaptive == baseline_pred_adaptive).sum()
                    stats[name][(n-n0, 4)]['c_info_gain'] = info_gain_adaptive
                    print(i, info_gain_adaptive/non_ignore_idx.sum())
                    print('adaptive baseline', (label_adaptive == baseline_pred_adaptive).sum()/non_ignore_idx.sum())
                pickle.dump(stats, open(f'{self.default_log_dir}/paper_logs/stats_with_h_{self.steps}.pkl', 'wb'))
                self.steps +=1

    def aggregated(self):
        self.steps = 0
        #self.model.eval()
        os.makedirs(f'{self.default_log_dir}/agg_stats/', exist_ok=True)

        agg_stats = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                if name not in agg_stats:
                    agg_stats[name] = {}
                else:
                    co = True
                    for n in self.n:
                        for max_size in self.max_sizes:
                            if (n, max_size) not in agg_stats[name]:
                                co = False
                    if co:
                        continue
                size = label.size()
                orig_label = label.cpu().numpy()
                flat_label = label.flatten().cpu().numpy()
                label_h = self.to_hierarchical(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                edge_map = self.get_edgemap(copy.deepcopy(label[0]).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL)    
                samples, samples_logits = None, None
                for n in tqdm(self.n, desc='ns'):
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1)
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]
                    samples_flattened = samples.reshape(samples.shape[0], -1) # (N, w*h)
                    samples_h = self.to_hierarchical(samples)
                    for max_size in self.max_sizes+['h']:
                        if (n, max_size) in agg_stats[name]:
                            continue
                        if max_size == 'h': 
                            sz = 'h'
                            s_shape = samples_h.shape; samples_flattened = samples_h.reshape(samples.shape[0], -1) # (N, w*h)
                            flat_label = label_h.flatten()
                            non_ignore_idx = flat_label != 255
                            max_size = 1
                            self.num_classes = 7+1
                            self.abstain_label = 7
                        else:
                            self.num_classes = 20
                            self.abstain_label = 19
                            flat_label = orig_label.flatten()
                            non_ignore_idx = flat_label != 255
                            sz = max_size
                        flat_label = flat_label[non_ignore_idx]
                        classes_certify_oh, pvals, radius, timings, cnt = self.run_certify(samples_flattened, n, self.tau, correction='bonferroni', greedy=self.greedy, max_size=max_size) # (w*h, 20)
                        classes_certify = classes_certify_oh.argmax(1)
                        classes_certify_oh = classes_certify_oh[non_ignore_idx]
                        classes_certify = classes_certify[non_ignore_idx]
                        two_idx = classes_certify_oh.sum(1) == max_size
                        non_abstain_idx = classes_certify != self.abstain_label
                        two_class_gt_oh = np.eye(classes_certify_oh.shape[-1])[flat_label[two_idx & non_abstain_idx]]
                        e = edge_map.flatten()[non_ignore_idx][two_idx & non_abstain_idx]

                        if (n, sz) not in agg_stats[name]:
                            agg_stats[name][(n, sz)] = {}
                        ei = edge_map.flatten()[non_ignore_idx][non_abstain_idx]
                        agg_stats[name][(n, sz)]['num_c_boundary_pixels'] = (ei==1).sum(0)
                        agg_stats[name][(n, sz)]['num_c_nonboundary_pixels'] = (ei==0).sum(0)

                        agg_stats[name][(n, sz)]['boundary_certified_pixels_@maxsize_count_per_class'] = two_class_gt_oh[e==1].sum(0)
                        agg_stats[name][(n, sz)]['nonboundary_certified_pixels_@maxsize_count_per_class'] = two_class_gt_oh[e==0].sum(0)
                        e = edge_map.flatten()[non_ignore_idx]
                        certified_acc_boundary, certified_acc_boundary_p, pos_boundary = self.certified_accuracy(classes_certify_oh[e==1], flat_label[e==1])
                        certified_acc_nonboundary, certified_acc_nonboundary_p, pose_nonboundary = self.certified_accuracy(classes_certify_oh[e==0], flat_label[e==0])
                        agg_stats[name][(n, sz)]['percentage_certified'] = (classes_certify != self.abstain_label).sum()/(len(classes_certify))
                        agg_stats[name][(n, sz)]['num_certified_pixels'] = (classes_certify != self.abstain_label).sum()
                        agg_stats[name][(n, sz)]['num_pixels'] = len(classes_certify)

                        agg_stats[name][(n, sz)]['certified_acc_boundary'] = certified_acc_boundary
                        agg_stats[name][(n, sz)]['certified_acc_nonboundary'] = certified_acc_nonboundary
                        agg_stats[name][(n, sz)]['certified_acc_boundary_p'] = certified_acc_boundary_p
                        agg_stats[name][(n, sz)]['certified_acc_nonboundary_p'] = certified_acc_nonboundary_p
                        agg_stats[name][(n, sz)]['pos_boundary_pixels'] = pos_boundary
                        agg_stats[name][(n, sz)]['pos_nonboundary_pixels'] = pose_nonboundary

                        agg_stats[name][(n, sz)]['certified_acc'], agg_stats[name][(n, sz)]['certified_acc_p'], agg_stats[name][(n, sz)]['pos'] = self.certified_accuracy(classes_certify_oh, flat_label)
                        agg_stats[name][(n, sz)]['certified_boundary_pixels_count'] = (classes_certify[e==1] != self.abstain_label).sum()
                        agg_stats[name][(n, sz)]['certified_nonboundary_pixels_count'] = (classes_certify[e==0] != self.abstain_label).sum()
                        agg_stats[name][(n, sz)]['num_boundary_pixels'] = (e==1).sum()
                        agg_stats[name][(n, sz)]['num_nonboundary_pixels'] = (e==0).sum()
                        if sz == 'h': max_size = None
                        _, strs, counts = self.get_string_set_histo(classes_certify_oh, max_size=max_size, h=sz=='h')
                        agg_stats[name][(n, sz)]['strs_histo'] = {k:v for k, v in zip(strs, counts)}
                        _, strs, counts = self.get_string_set_histo(classes_certify_oh[classes_certify != self.abstain_label], max_size=max_size, h=sz=='h')
                        agg_stats[name][(n, sz)]['certified_strs_histo'] = {k:v for k, v in zip(strs, counts)}
                        agg_stats[name][(n, sz)]['label_classes_freq'] = np.eye(classes_certify_oh.shape[-1])[flat_label].sum(0)
                pickle.dump(agg_stats, open(f'{self.default_log_dir}/agg_stats/agg_stats_{self.steps}.pkl', 'wb'))
                self.steps +=1
    
    def gen_cached_samples(self):
        self.steps = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                size = label.size()
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                #if os.path.exists(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_logits.pkl'):
                #    continue
                os.makedirs(f'{self.default_log_dir}/cached_samples/', exist_ok=True)
                self.log(f'Generating {self.max_n} samples for {name}')
                _, logits = self.sample(image_np01, size, 500, sigma = self.sigma, posteriors=True)
                pickle.dump(logits, open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_logits.pkl', 'wb'), protocol=4)
                self.steps +=1

    def test_certification_jump(self):
        self.steps = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):
                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gtFine_labelIds', '')
                size = label.size()
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                samples, samples_logits = None, None
                for n in tqdm(self.n, desc='ns'):
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1)
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]
                    s_shape = samples.shape # (N, 512, 1024)
                    samples_flattened = samples.reshape(samples.shape[0], -1) # (N, w*h)
                    for max_size in self.max_sizes:
                        classes_certify_oh, pvals, radius, timings, cnt = self.run_certify(samples_flattened, n, self.tau, correction='bonferroni', greedy=self.greedy, max_size=max_size) # (w*h, 20)
                        classes_certify = classes_certify_oh.argmax(1)
                        greedy_percentage_certified = (classes_certify != self.abstain_label).sum()/(len(classes_certify))
                        strings_histo, strs, counts = self.get_string_set_histo(classes_certify_oh, max_size=max_size) # (3, 512, 1024)

    def dataset_certify(self):
        self.log('Certify start')
        # self.model.eval()

        self.steps = 0
        overall_confusion_matrix = None
        overall_baseline_acc = 0
        only_pc = True
        strs_dict = {}
        m_percentage_certified = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):

                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                if len(self.args.NN) > 0 and self.steps not in self.args.NN: continue

                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gt_labelIds', '')
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                label = label[0]
                size = label.size()
                edge_map = self.get_edgemap(copy.deepcopy(label).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                baseline_pred, logits = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
                baseline_pred = baseline_pred[0]

                if self.hierarchical: label = self.to_hierarchical(label.unsqueeze(0)).cpu().squeeze().numpy()
                if self.hierarchical: baseline_pred = self.to_hierarchical(np.expand_dims(baseline_pred, 0)).squeeze()

                confusion_matrix, baseline_acc = self.get_confusion_matrix(label, baseline_pred, self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                if overall_confusion_matrix is None: overall_confusion_matrix = confusion_matrix
                else: overall_confusion_matrix += confusion_matrix

                cm_rgb = self.graph_confusion_matrix(confusion_matrix, title=f'Baseline Accuracy={str(baseline_acc)}', size=(image.shape[3], image.shape[2]))
                self.steps +=1

                gt_palettle = self.get_seg_palette(label)
                baseline_pred_palette = self.get_seg_palette(baseline_pred)
                ims = torch.cat([torch.tensor(image).byte(),
                                torch.tensor(gt_palettle),
                                torch.tensor(cm_rgb).unsqueeze(0),
                                torch.tensor(baseline_pred_palette)])
                log_img = torchvision.utils.make_grid(ims, nrow=2)
                self.writer.add_image('Baseline', log_img, self.steps)
                if self.baseline:
                    continue
                # baseline_acc = self.get_acc(label, baseline_pred, size)

                # baseline_posteriors = F.softmax(torch.tensor(logits), dim=1).cpu().numpy() # (1, 19, 1024, 2048)

                # if self.hierarchical: baseline_pred = to_hierarchical(baseline_pred)
                # if self.smoothing: baseline_pred =  ModePool2d(kernel_size=self.smooth_kernel_size)(torch.tensor(baseline_pred).unsqueeze_(0)).squeeze_(0).cpu().numpy()

                # gt_palettle = self.get_seg_palette(label)
                # baseline_pred_palette = self.get_seg_palette(baseline_pred)

                samples, samples_logits = None, None
                greedy_percentage_certified_ls = []
                for n in tqdm(self.n, desc='ns'):
                    # if samples is not None:
                    #     samples = samples[:n]
                    #     if samples_logits is not None:
                    #         samples_logits = samples_logits[:n]
                    # else:
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1)
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]

                    if self.hierarchical: samples = self.to_hierarchical(samples)

                    # posteriors = F.softmax(torch.tensor(samples_logits), dim=1).cpu().numpy() # (300, 19, 512, 1024)
                    # diff, posteriors_m_std = self.get_difference(posteriors)
                    # rgb_mean_std_posteriors = self.overlay_heatmap(posteriors_m_std, image_np01*255)
                    # rgb_diff = self.overlay_heatmap(1-diff, image_np01*255)

                    if not self.certify:
                        continue
                    s_shape = samples.shape # (N, 512, 1024)
                    # flatten samples
                    samples_flattened = samples.reshape(samples.shape[0], -1) # (N, w*h)

                    self.log(f'Certifying {name} at n={n}')
                    if self.greedy:
                        ls = []
                        for max_size in self.max_sizes:
                            classes_certify_oh, pvals, radius, timings, cnt = self.run_certify(samples_flattened, n, self.tau, correction='bonferroni', greedy=self.greedy, max_size=max_size) # (w*h, 20)
                            classes_certify = classes_certify_oh.argmax(1)
                            greedy_percentage_certified = (classes_certify != self.abstain_label).sum()/(len(classes_certify))
                            greedy_percentage_certified_ls.append(greedy_percentage_certified)
                            strings_histo, strs, counts = self.get_string_set_histo(classes_certify_oh, max_size=max_size) # (3, 512, 1024)
                            ls.append(torch.tensor(strings_histo).unsqueeze(0))


                            if n == 300:
                                if max_size not in strs_dict:
                                    strs_dict[max_size] = {}
                                for s, c in zip(strs, counts):
                                    if s not in strs_dict[max_size]:
                                        strs_dict[max_size][s] = 0
                                    strs_dict[max_size][s] += c
                                # pickle.dump(strs_dict, open(f'{self.default_log_dir}/cached_stats/strs_dict_acc_{self.steps}.pkl', 'wb'))
                                if name not in m_percentage_certified:
                                    m_percentage_certified[name] = []
                                m_percentage_certified[name].append(greedy_percentage_certified)
                        #print(f'Writing {self.default_log_dir}/cached_stats/percentage_certified_{self.steps}.pkl')
                        #pickle.dump(m_percentage_certified, open(f'{self.default_log_dir}/cached_stats/percentage_certified_{self.steps}.pkl', 'wb'))
                        # log_img = torchvision.utils.make_grid(torch.cat(ls))
                        #self.writer.add_image(f'{name}-ClassSet', log_img, n)
                        if only_pc and 0:
                            continue

                        # greedy_conf_matrix, greedy_certified_accuracy = self.get_confusion_matrix(label, np.reshape(classes_certify, s_shape[1:]), self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                        strings_histo, strs, counts = self.get_string_set_histo(classes_certify_oh) # (3, 512, 1024)
                        class_set_sizes = classes_certify_oh[:, 0:19].sum(1) #(w*h)
                        sizes_histogram = self.plot_histogram(class_set_sizes) 
                        # boundary vs nonboundary
                        boundary_sizes_histogram = self.plot_histogram(class_set_sizes[np.reshape(edge_map, -1) == 1], title='Boundary')
                        nonboundary_sizes_histogram = self.plot_histogram(class_set_sizes[np.reshape(edge_map, -1) == 0], title='Non-boundary')

                        boundary_stack_histgoram = self.plot_histogram_boundary(edge_map, class_set_sizes)
                        class_set_sizes = np.reshape(class_set_sizes, s_shape[1:]) # (512, 1024)
                        set_sizes_heatmap = self.overlay_heatmap(class_set_sizes, image_np01*255)
                        classes_certify_im = np.reshape(classes_certify, s_shape[1:]) # (512, 1024)
                        classes_certify_palette = self.get_seg_palette(classes_certify_im)
                        classes_certify_palette = np.array(cv2.putText(classes_certify_palette.squeeze().transpose(1, 2, 0), 
                                                              f'%certified={str(greedy_percentage_certified.round(3)*100)}',\
                                                                        (60,60), 
                                                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                                                        2, (0, 0, 0), 3, 
                                                                        cv2.LINE_AA)).transpose(2, 0, 1)
                        
                        edge_map = np.stack((edge_map,)*3, axis=-1).transpose(2, 0, 1).astype(np.uint8)*255
                        ls = []
                        # (1, 3, 512, 1024)
                        ls.append(torch.tensor(set_sizes_heatmap).unsqueeze(0))
                        ls.append(torch.tensor(strings_histo).unsqueeze(0).byte())

                        ls.append(torch.tensor(classes_certify_palette).unsqueeze(0))
                        ls.append(torch.tensor(sizes_histogram.astype(np.uint8)).unsqueeze(0))
                        ls.append(torch.tensor(edge_map).unsqueeze(0))
                        ls.append(torch.tensor(boundary_stack_histgoram.astype(np.uint8)).unsqueeze(0))

                        ls.append(torch.tensor(boundary_sizes_histogram.astype(np.uint8)).unsqueeze(0))
                        ls.append(torch.tensor(nonboundary_sizes_histogram.astype(np.uint8)).unsqueeze(0))                        

                        ims = torch.cat(ls)
                        log_img = torchvision.utils.make_grid(ims, nrow=2)
                        self.writer.add_image(f'{name}', log_img, n)
                    # classes_certify_non_greedy,_, _, _, _= self.run_certify(samples_flattened, n, self.tau, correction='bonferroni', greedy=False)
                    # certified_conf_matrix, certified_accuracy = self.get_confusion_matrix(label, np.reshape(classes_certify_non_greedy, s_shape[1:]), self.num_classes, self.config.TRAIN.IGNORE_LABEL)

                    # percentage_certified = (classes_certify_non_greedy != self.abstain_label).sum()/classes_certify_non_greedy.shape[0]
                    # classes_certify_non_greedy = np.reshape(classes_certify_non_greedy, s_shape[1:]) 
                    # classes_certify_non_greedy_palette = self.get_seg_palette(classes_certify_non_greedy)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # classes_certify_non_greedy_palette = cv2.putText(classes_certify_non_greedy_palette.squeeze().transpose(1, 2, 0), f'%certified={str(percentage_certified.round(3))}',\
                    #                                                 (60,60), font, 2, (0, 0, 0), 3, cv2.LINE_AA)

                    # time_pvals, time_correction = timings
                    # classes_certify = np.reshape(classes_certify, s_shape[1:]) # (1, 1024, 2048)
                    # pvals = np.reshape(pvals, s_shape[1:]) # (1, 1024, 2048)
                    # classes_certify_palette = self.get_seg_palette(classes_certify)
                    # confusion_matrix, cert_acc = self.get_confusion_matrix(label, classes_certify, self.num_classes, self.config.TRAIN.IGNORE_LABEL)

                    # # logging
                    # ls = [torch.tensor(image.clone().detach().cpu().numpy()).byte(),
                    #                     torch.tensor(gt_palettle), \
                    #                     torch.tensor(rgb_diff.astype(np.uint8).transpose(2, 0, 1)).unsqueeze(0),
                    #                     torch.tensor(baseline_pred_palette), \
                    #                     torch.tensor(rgb_mean_std_posteriors.astype(np.uint8).transpose(2, 0, 1)).unsqueeze(0),
                    #                     torch.tensor(classes_certify_non_greedy_palette.astype(np.uint8).transpose(2, 0, 1)).unsqueeze(0)]
                    # if self.greedy:
                    #     ls.append(torch.tensor(set_sizes_heatmap.astype(np.uint8).transpose(2, 0, 1)).unsqueeze(0))
                    #     ls.append(torch.tensor10(classes_certify_palette))
                    #     ls.append(torch., 2, 3tensor(sizes_histogram.astype(np.uint8).transpose(2, 0, 1)).unsqueeze(0))
                    #     ls.append(torch.tensor(strings_histo).unsqueeze(0))

                    # # print(cnt.shape, cnt.max())
                    # ims = torch.cat(ls)
                    # log_img = torchvision.utils.make_grid(ims, nrow=2)
                    # self.writer.add_image(f'{name}', log_img, n)

                self.steps +=1
                cpr_max_size_line_graph = self.plot_cpr_max_size(greedy_percentage_certified_ls, self.max_sizes, self.n)
                ls = [torch.tensor(cpr_max_size_line_graph).unsqueeze(0)]
                ims = torch.cat(ls)
                log_img = torchvision.utils.make_grid(ims)
                self.writer.add_image(f'{name} Overall', log_img, self.steps)
        print('Baseline_+CM')
        heatmap_rgb = self.graph_confusion_matrix(np.log(overall_confusion_matrix), title=f'Accuracy={self.cm_to_acc(overall_confusion_matrix)}', size=None)
        log_img = torchvision.utils.make_grid(torch.tensor(heatmap_rgb).unsqueeze(0))
        self.writer.add_image('Baseline-CM', log_img, 1)
        self.writer.add_image('Baseline-CM', log_img, 1)
        ls = []
        for set_size, str_d in strs_dict.items():
            # sort in descending order
            sorted_dict = dict(sorted(str_d.items(), key=lambda x:x[1], reverse=True))
            str_overall_im = self.plot_bar(list(sorted_dict.keys()), np.log(np.array(list(sorted_dict.values()))))
            ls.append(torch.tensor(str_overall_im).unsqueeze(0))
        #  examine traffic light
        for set_size, str_d in strs_dict.items():
            sorted_dict = dict(sorted(str_d.items(), key=lambda x:x[1], reverse=True))
            x, y = [], []
            for k, v in sorted_dict.items():
                if 'traffic sign' in k:
                    x.append(k)
                    y.append(v)
            str_overall_im = self.plot_bar(x, np.log(np.array(y)))
            ls.append(torch.tensor(str_overall_im).unsqueeze(0))

        log_img = torchvision.utils.make_grid(torch.cat(ls), nrow=3)
        self.writer.add_image(f'Overall-ClassSet', log_img, 0)
        self.writer.add_image(f'Overall-ClassSet', log_img, 0)

                    # self.writer.add_image('image', image/255.0, self.steps)

    def dataset_certify_h(self):
        self.log('Certify start')
        self.model.eval()
        self.steps = 0
        overall_confusion_matrix = None
        overall_baseline_acc = 0
        only_pc = True
        strs_dict = {}
        m_percentage_certified = {}
        m_certified_acc = {}
        with torch.no_grad():
            for batch in tqdm(self.testloader, desc='Main Loop'):

                if self.steps < self.args.N0: self.steps +=1; continue
                if self.steps >= self.args.N: break
                if len(self.args.NN) > 0 and self.steps not in self.args.NN: continue

                image, label, _, name = batch  #(1, 3, 1024, 2048), (1, 1024, 2048)
                name = name[0].replace('_gt_labelIds', '')
                image_np01 = image[0].numpy().transpose((1, 2, 0)).astype(np.float32).copy()/255.0 # (1024, 2048, 3)
                label = label[0]
                size = label.size()
                edge_map = self.get_edgemap(copy.deepcopy(label).cpu().numpy(), self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                baseline_pred, logits = self.sample(image_np01, size, n=1, sigma=0, posteriors=True) # (1, 512, 1024)
                baseline_pred = baseline_pred[0]

                confusion_matrix, baseline_acc = self.get_confusion_matrix(label, baseline_pred, self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                if overall_confusion_matrix is None: overall_confusion_matrix = confusion_matrix
                else: overall_confusion_matrix += confusion_matrix

                cm_rgb = self.graph_confusion_matrix(confusion_matrix, title=f'Baseline Accuracy={str(baseline_acc)}', size=(image.shape[3], image.shape[2]))
                self.steps +=1

                gt_palettle = self.get_seg_palette(label)
                baseline_pred_palette = self.get_seg_palette(baseline_pred)
                ims = torch.cat([torch.tensor(image).byte(),
                                torch.tensor(gt_palettle),
                                torch.tensor(cm_rgb).unsqueeze(0),
                                torch.tensor(baseline_pred_palette)])
                log_img = torchvision.utils.make_grid(ims, nrow=2)
                # self.writer.add_image('Baseline', log_img, self.steps)
                if self.baseline:
                    continue


                label_orig = label
                label_h = self.to_hierarchical(torch.tensor(label).clone().detach().unsqueeze(0)).cpu().squeeze().numpy()

                samples, samples_logits = None, None
                greedy_percentage_certified_ls = []
                greedy_certified_acc_ls = []
                for n in tqdm(self.n, desc='ns'):
                    if samples_logits is None:
                        samples_logits = self.get_samples(image_np01, n, size, name)
                        samples = samples_logits.argmax(1)
                    else:
                        samples, samples_logits = samples[0:n], samples_logits[0:n]

                    samples_h = self.to_hierarchical(samples)
                    label = label_orig
                    s_shape = samples.shape # (N, 512, 1024)
                    samples_flattened = samples.reshape(samples.shape[0], -1) # (N, w*h)
                    self.log(f'Certifying {name} at n={n}')
                    ls = []
                    sz = None
                    for max_size in tqdm(self.max_sizes + ['h']):
                        if max_size == 'h': 
                            sz = 'h'
                            s_shape = samples_h.shape; samples_flattened = samples_h.reshape(samples.shape[0], -1) # (N, w*h)
                            label = label_h
                            max_size = 1
                            self.num_classes = 7+1
                            self.abstain_label = 7
                        else:
                            self.num_classes = 20
                            self.abstain_label = 19
                            label = label
                            sz = max_size 
                        classes_certify_oh, pvals, radius, timings, cnt = self.run_certify(samples_flattened, n, self.tau, correction='bonferroni', greedy=self.greedy, max_size=max_size) # (w*h, 20)
                        classes_certify = classes_certify_oh.argmax(1)
                        greedy_percentage_certified = (classes_certify != self.abstain_label).sum()/(len(classes_certify))
                        greedy_percentage_certified_ls.append(greedy_percentage_certified)
                        classes_certify_im = np.reshape(classes_certify, s_shape[1:]) # (512, 1024)
                        certified_acc = self.certified_accuracy(classes_certify_oh, label)
                        # _, certified_acc = self.get_confusion_matrix(label, classes_certify_im, self.num_classes, self.config.TRAIN.IGNORE_LABEL)
                        greedy_certified_acc_ls.append(certified_acc)
                        classes_certify_palette = self.write_on_palette(self.get_seg_palette(classes_certify_im), 
                                                                        f'S{sz},%certified={str(np.round(greedy_percentage_certified, 3)*100)},acc={str(np.round(certified_acc, 3)*100)}')
                        ls.append(torch.tensor(classes_certify_palette).unsqueeze(0))                        
                        if n == 300:
                            if name not in m_percentage_certified:
                                m_percentage_certified[name] = []
                            if name not in m_certified_acc:
                                m_certified_acc[name] = []
                            m_percentage_certified[name].append((sz, greedy_percentage_certified))
                            m_certified_acc[name].append((sz, certified_acc))

                    print(f'Writing {self.default_log_dir}/cached_stats/m_percentage_certified_{self.steps}.pkl')
                    print(f'Writing {self.default_log_dir}/cached_stats/m_certified_acc_{self.steps}.pkl')

                    ims = torch.cat([torch.tensor(image).byte(),
                    torch.tensor(gt_palettle)]+ls)
                    log_img = torchvision.utils.make_grid(ims, nrow=2)
                    self.writer.add_image(f'{name}', log_img, n)
                cpr_max_size_line_graph = self.plot_cpr_max_size(greedy_percentage_certified_ls, self.max_sizes+['h'], self.n)
                cacc_max_size_line_graph = self.plot_cpr_max_size(greedy_certified_acc_ls, self.max_sizes+['h'], self.n, title='Certified Accuracy vs. n', y_title='Certified Accuracy')

                ls = [torch.tensor(cpr_max_size_line_graph).unsqueeze(0), torch.tensor(cacc_max_size_line_graph).unsqueeze(0)]
                ims = torch.cat(ls)
                log_img = torchvision.utils.make_grid(ims)
                self.writer.add_image(f'{name} cpr cacc', log_img, self.steps)

    def oh_to_string_set(self, oh, h=False):
        if h: labels_dict = self.labels_dict_h
        else: labels_dict = self.labels_dict
        strings = np.array(list(labels_dict.values()))
        if (np.array(oh)==1).sum() == 0:
            return 'abstain'
        return ','.join(list(strings[0:20][np.array(oh)==1]))
    
    def get_string_set_histo(self, classes_certify, max_size=None, h=False):
        import pandas as pd
        df = pd.DataFrame(classes_certify)
        df['combined'] = df.values.tolist()
        df = df.loc[:, ['combined']]
        df['combined'] = df['combined'].transform(lambda x: self.oh_to_string_set(x, h))
        new_df = df.groupby(['combined'])['combined'].count()
        groups, counts = list(new_df.index), list(new_df.values)
        counts, groups = zip(*sorted(zip(counts, groups)))
        x, y = list(reversed(list(groups))), list(reversed(list(counts)))
        if max_size is not None and max_size != 1 and max_size != 19:
            x_, y_ = [], []
            for i, x__ in enumerate(x):
                if x__.count(',') == (max_size - 1):
                    x_.append(x[i])
                    y_.append(y[i])
            x, y = x_, y_

        #rgb = self.plot_bar(x[0:20], np.log(np.array(y))[0:20])
        return None, x, y
    
    def plot_bar(self, x, y, size=(1024, 512)):
        fig = go.Figure([go.Bar(x=x, y=y)])
        # add colorbar
        image_bytes = fig.to_image(format="png")
        rgb = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        if size is not None:
            rgb = cv2.resize(rgb, size)
        rgb = np.array(rgb).transpose(2, 0, 1)[0:3, :, :].astype(np.uint8)
        return rgb
    
    def write_on_palette(self, im, txt):
        im = np.array(cv2.putText(im.squeeze().transpose(1, 2, 0), 
                                                txt,\
                                                        (60,60), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                                        2, (0, 0, 0), 2, 
                                                        cv2.LINE_AA)).transpose(2, 0, 1)
        return im
    
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

    def plot_histogram_boundary(self, edge_map, class_set_sizes, size=(1024, 512)):
        boundary = class_set_sizes[np.reshape(edge_map, -1)==1]
        nonboundary = class_set_sizes[np.reshape(edge_map, -1)==0]
        boundary_dict = collections.Counter(boundary)
        nonboundary_dict = collections.Counter(nonboundary)

        boundary_dict = dict(sorted(boundary_dict.items(), key=lambda x:x[1], reverse=True))
        nonboundary_dict = dict(sorted(nonboundary_dict.items(), key=lambda x:x[1], reverse=True))
        sizes = list(set(list(boundary_dict.keys())+list(nonboundary_dict.keys())))
        y_boundary, y_nonboundary = [], []
        for sz in sizes:
            if sz in boundary_dict:
                y_boundary.append(boundary_dict[sz]/len(class_set_sizes))
            else:
                y_boundary.append(0)
            
            if sz in nonboundary_dict:
                y_nonboundary.append(nonboundary_dict[sz]/len(class_set_sizes))
            else:
                y_nonboundary.append(0)

        fig = go.Figure(data=[
            go.Bar(name='Boundary %', x=sizes, y=y_boundary),
            go.Bar(name='Non-boundary %', x=sizes, y=y_nonboundary)
        ])
        fig.update_layout(width=size[0], height=size[1], margin=dict(l=20, r=20, t=20, b=20), barmode='stack')

        img_bytes = fig.to_image(format="png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        rgb_img = np.array(img)
    
        # Return the RGB image of the histogram
        return rgb_img.transpose(2, 0, 1).astype(np.uint8)
    
    def plot_histogram(self, arr, bins=19, size=(1024, 512), title=''):
        # Create a histogram of the input array        
        # Create a Plotly figure with a histogram
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Histogram(x=arr, histnorm='percent'), row=1, col=1)
        
        # Get the Plotly figure as an image and convert to RGB
        fig.update_layout(width=size[0], height=size[1], margin=dict(l=20, r=20, t=20, b=20), title=title)
        img_bytes = fig.to_image(format="png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        rgb_img = np.array(img)
    
        # Return the RGB image of the histogram
        return rgb_img.transpose(2, 0, 1).astype(np.uint8)
    
    def overlay_heatmap(self, heatmap, image, alpha=0.5):
        heatmap_norm = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        rgb = cv2.addWeighted(image[...,::-1].astype(np.uint8), 1-alpha, heatmap_color.astype(np.uint8), alpha, 0)[...,::-1]
        return np.array(rgb).transpose(2, 0, 1).astype(np.uint8)
    
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
    
    def get_edgemap(self, label, num_classes, ignore, filer_size=5):
        label_ = label
        label_[label == ignore] = num_classes
        label_ = np.uint8(label_/label_.max()*255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filer_size, filer_size))
        edge_map = np.uint8(cv2.dilate(cv2.Canny(label_, 0, 100), kernel))
        return np.uint8(edge_map/255.0)
    
    def plot_cpr_max_size(self, greedy_percentage_certified_ls, max_sizes, ns, title='% Certified', y_title='Percentage Certified Pixels', size=None):
        certified = np.array(np.array(greedy_percentage_certified_ls).reshape(len(ns), len(max_sizes)))
        fig = go.Figure()

        for i, max_size in enumerate(max_sizes):
            x = ns
            y = certified[:, i]
            fig.add_trace(go.Scatter(x=list(reversed(x)), y=list(reversed(y)),
                                mode='lines+markers',
                                name=f'max_set_size={max_size}'))
        
        fig.update_layout(
            title=title,
            xaxis_title='number of samples (n)',
            yaxis_title = y_title
        )
        image_bytes = fig.to_image(format="png")
        rgb = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        if size is not None:
            rgb = cv2.resize(rgb, size)
        return rgb.transpose(2, 0, 1)[0:3, :, :]

    def graph_confusion_matrix(self, confusion_matrix, title='', size=(1024, 512)):
        labels = [self.labels_dict[i] for i in range(confusion_matrix.shape[0])]
        conf_matrix_text = [[str(y) for y in x] for x in confusion_matrix]
        fig = go.Figure(data=go.Heatmap(z=confusion_matrix, text=conf_matrix_text,
                    x=labels, y=labels, colorscale='Viridis'))
        fig.update_layout(
            title=title,
            xaxis_title='Predicted Value',
            yaxis_title = 'Ground Truth'
        )
        # add colorbar
        image_bytes = fig.to_image(format="png")
        rgb = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        if size is not None:
            rgb = cv2.resize(rgb, size)
        return rgb.transpose(2, 0, 1)[0:3, :, :]

    def log_posteriors_insights(self, classes_certify, n, logits, cnt, pvals, name):
        self.log('logging posteriors')
        posteriors = F.softmax(torch.tensor(logits), dim=1).cpu().numpy()
        classes_certify = classes_certify.reshape(1024*2048)
        top_count = np.array(cnt).reshape(1024*2048)
        pvals = np.array(pvals).reshape(1024*2048)
        abstain_idcs = classes_certify == (self.num_classes-1)
        certify_idcs = classes_certify != (self.num_classes-1)

        abstain_cnt = top_count[abstain_idcs]
        certify_cnt = top_count[certify_idcs]
        abstain_pvals = pvals[abstain_idcs]
        certify_pvals = pvals[certify_idcs]
        try:
            if len(certify_cnt) > 0:

                self.writer.add_scalar(f"{name} (C) Mean top-ratio:", np.mean(certify_cnt)/n, n)
                self.writer.add_scalar(f"{name} (C) Min top-ratio:", np.min(certify_cnt)/n, n)
                self.writer.add_scalar(f"{name} (C) Max top-ratio:", np.max(certify_cnt)/n, n)
                self.writer.add_histogram(f'{name} (C) TCR Histogram',  certify_cnt/n, n)
                self.writer.add_histogram(f'{name} (C) pvals Histogram',  certify_pvals, n)
                tcr1 = certify_cnt[0]/n

        except:
            print(f'No certified idcs at m={n}')

        self.writer.add_scalar(f"{name} (A) Mean top-ratio:", np.mean(abstain_cnt)/n, n)
        self.writer.add_scalar(f"{name} (A) Min top-ratio:", np.min(abstain_cnt)/n, n)
        self.writer.add_scalar(f"{name} (A) Max top-ratio:", np.max(abstain_cnt)/n, n)

        self.writer.add_histogram(f'{name} (A) TCR Histogram',  abstain_cnt/n, n)

        self.writer.add_histogram(f'{name} all-pvals Histogram',  pvals, n)
        self.writer.add_histogram(f'{name} (A) pvals Histogram',  abstain_pvals, n)


        mean_posteriors = posteriors.mean(axis=0).reshape(19, 1024*2048)
        certified_posteriors = mean_posteriors[:, certify_idcs]
        abstain_porteriors = mean_posteriors[:, abstain_idcs]
        tcr2 = np.max(abstain_cnt)/n
        tcr3 = abstain_cnt[np.isclose(abstain_cnt, [np.mean(abstain_cnt)], atol=3)][0]/n
        tcr4 = np.min(abstain_cnt)/n

        for i in range(self.num_classes-1):
            try:
                if len(certify_cnt) > 0:
                    tcr1 = certify_cnt[0]/n
                    if len(certified_posteriors) > 0:
                        self.writer.add_scalar(f'{name} (C) Pixel Posteriors tcr={tcr1.round(2)} (max) n={n}', certified_posteriors[i, 0], i)
            except:
                x=None
            self.writer.add_scalar(f'{name} (A) Pixel Posteriors tcr={tcr2.round(2)} (max) n={n}', abstain_porteriors[:, abstain_cnt == np.max(abstain_cnt)][i, 0], i)
            self.writer.add_scalar(f'{name} (A) Pixel Posteriors tcr={tcr3.round(2)} (mean) n={n}', abstain_porteriors[:, np.isclose(abstain_cnt, [np.mean(abstain_cnt)], atol=3)][i, 0], i)
            self.writer.add_scalar(f'{name} (A) Pixel Posteriors tcr={tcr4.round(2)} (min) n={n}',  abstain_porteriors[:, abstain_cnt == np.min(abstain_cnt)][i, 0], i)


    def get_samples(self, image_np01, n, size, name):
        try:
            #samples = pickle.load(open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_classes.pkl', 'rb'))
            logits = pickle.load(open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_logits.pkl', 'rb'))
            self.log(f'Loading cached sample {name}')
        except:
            self.log(f'Generating {self.max_n} samples for {name}')
            _, logits = self.sample(image_np01, size, self.max_n, sigma = self.sigma, posteriors=True)
            pickle.dump(logits, open(f'{self.default_log_dir}/cached_samples/{name}_{self.max_n}_logits.pkl', 'wb'), protocol=4)
        return logits[:n]


    def smooth(self, samples):
        bs = 16
        s_i = []
        for i in tqdm(range(0, samples.shape[0], bs), desc='ModePool2D'):
            s_i.append(ModePool2d(kernel_size=self.smooth_kernel_size)(torch.tensor(samples[i:i+bs]).unsqueeze_(1)).squeeze_(1).cpu().numpy())
        s = np.concatenate(s_i)

    def sample(self, image_np01, size, n, sigma, posteriors=False, do_tqdm=False):
        BS = self.config.TEST.BATCH_SIZE_PER_GPU
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
                            unscaled=False) # torch.Size([1, 19, 1024, 2048])
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
            return np.concatenate(out), np.concatenate(out_pred)
        return np.concatenate(out), None
    
    def acc_from_conf(self, confusion_matrix):
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        return pixel_acc
    
    def get_acc(self, label, baseline_pred, size):
        confusion_matrix = get_confusion_matrix(label, baseline_pred, size, 
                                                self.num_classes, 
                                                self.config.TRAIN.IGNORE_LABEL)
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        return pixel_acc
    
    def get_confusion_matrix(self, label, seg_pred, num_class, ignore=-1):
        """
        Cal cute the confusion matrix by given label and pred
        """
        seg_gt = np.asarray(label, dtype=int)

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
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        return confusion_matrix, pixel_acc
    
    def cm_to_acc(self, confusion_matrix):
        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        pixel_acc = tp.sum()/pos.sum()
        return pixel_acc
    
    def get_seg_palette(self, label):
        img = np.asarray(label, dtype=np.uint8)
        I = (img == self.abstain_label)
        img = self.test_dataset.convert_label(img, inverse=True) # converts to id
        img[I] = self.abstain_mapping
        img = Image.fromarray(img)
        img.putpalette(self.palette)
        img.save(f'{self.default_log_dir}/tmp.png')
        seg_rgb = cv2.imread(f'{self.default_log_dir}/tmp.png')[...,::-1]
        seg_pallete = np.expand_dims(np.array(seg_rgb).transpose(2, 0, 1), 0)
        del img
        return seg_pallete
    
    def process_label(self, label):
        return label
    
    def log(self, msg):
        print(msg)
    
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    remote = False
    if remote:
        mp.spawn(
            Certifier,
            nprocs=world_size,
            args=(world_size,),)
    else:
        trainer = Certifier(1, 1)