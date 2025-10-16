import os
import random
import numpy as np
import torch

from models.MCCFN import MCCFN

# 消融实验，MCCFN_1：单一尺度、MCCFN_2：普通卷积、MCCFN_3：没有注意力
from models.Ablation_study.Single_Scale import MCCFN_1
from models.Ablation_study.Ordinary_convolution import MCCFN_2
from models.Ablation_study.Lack_of_attention import MCCFN_3



import yaml
import logging
from datetime import datetime



def fix_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_log_dir_index(out_dir):
    dirs = [x[0] for x in os.listdir(out_dir)]
    if '.' in dirs:  # minor change for .ipynb
        dirs.remove('.')
    log_dir_index = '_' + str(len(dirs))

    return log_dir_index


def merge_args2cfg(cfg, args_dict):
    for k, v in args_dict.items():
        setattr(cfg, k, v)
    return cfg



def create_logger(filename, file_handle=True):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger

def log_exp_settings(logger, cfg):
    """
    log the current experiment settings.
    """
    logger.info('=' * 20)
    log_dict = cfg.__dict__.copy()
    for k, v in log_dict.items():
        logger.info(f'{k} : {v}')
    logger.info('=' * 20)

def load_model(cfg):
    if cfg.model == "MCCFN":
        model = MCCFN(11, 128, 36, 512, 2, [36,48,64,128, 256]).to(cfg.device)
        return model

    elif cfg.model == "MCCFN_1":
        model = MCCFN_1(11, 128, 36, 512, 2, [36,48,64,128, 256]).to(cfg.device)
        return model
    elif cfg.model == "MCCFN_2":
        model = MCCFN_2(11, 128, 36, 512, 2, [36,48,64,128, 256]).to(cfg.device)
        return model
    elif cfg.model == "MCCFN_3":
        model = MCCFN_3(11, 128, 36, 512, 2, [36,48,64,128, 256]).to(cfg.device)
        return model
    else:
        raise ValueError(f"Unknown Model: {cfg.model}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class Config:
    def __init__(self, dataset, model, train=True):
        self.dataset = dataset
        self.model = model
        yaml_name = './config/%s.yml' % dataset
        if not os.path.exists(yaml_name):
            raise NotImplementedError(f"can not find cfg file: {yaml_name}")
        cfg = yaml.safe_load(open(yaml_name, 'r'))

        self.base_dir = 'train' if train else 'inference'
        os.makedirs(self.base_dir, exist_ok=True)

        self.epochs = cfg['epochs']
        self.batch_size = cfg['batch_size']
        self.patience = cfg['patience']
        self.milestone_step = cfg['milestone_step']
        self.gamma = cfg['gamma']
        self.lr = cfg['lr']
        self.acc_threshold = cfg['acc_threshold']

        self.initial_dropout_rate = cfg['initial_dropout_rate']
        self.max_dropout_rate = cfg['max_dropout_rate']
        self.dropout_threshold = cfg['dropout_threshold']
        self.patience_acc_diff = cfg['patience_acc_diff']

        self.num_classes = cfg['num_classes']
        self.num_level = cfg['num_level']
        self.kernel_size = cfg['kernel_size']
        self.in_channels = cfg['in_channels']
        self.latent_dim = cfg['latent_dim']
        self.monitor = cfg['monitor']
        self.test_batch_size = cfg['test_batch_size']

        if self.dataset == '2016.10a':
            self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                            b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9, b'AM-SSB': 10}
        elif dataset == '2016.10b':
            self.classes = {b'QAM16': 0, b'QAM64': 1, b'8PSK': 2, b'WBFM': 3, b'BPSK': 4,
                            b'CPFSK': 5, b'AM-DSB': 6, b'GFSK': 7, b'PAM4': 8, b'QPSK': 9}
        elif dataset == '2016.04c':
            self.classes = {b'8PSK': 0, b'AM-DSB': 1, b'AM-SSB': 2, b'BPSK': 3, b'CPFSK': 4,
                       b'GFSK': 5, b'PAM4': 6, b'QAM16': 7, b'QAM64': 8, b'QPSK': 9, b'WBFM': 10}
        elif dataset == 'rml22':
            self.classes = {'QAM16': 0, 'QAM64': 1, '8PSK': 2, 'WBFM': 3, 'BPSK': 4, 'CPFSK': 5, 'AM-DSB': 6, 'GFSK': 7,
              'PAM4': 8, 'QPSK': 9, 'AM-SSB': 10}
        else:
            raise NotImplementedError(f'Not Implement dataset:{self.dataset}')

        # 获取当前日期和时间，格式化为 "年_月_日_时_分_秒" 或者 "年/月/日/时/分/秒"
        current_time = datetime.now().strftime("%H_%M_%S")  # 使用下划线分隔
        current_day = datetime.now().strftime("%Y_%m_%d")  # 使用下划线分隔
        # 如果希望使用斜线，可以使用 "%Y/%m/%d/%H/%M/%S" 替代



        self.model_log_dir = '%s/%s' % (self.base_dir, self.model)
        os.makedirs(self.model_log_dir, exist_ok=True)

        self.current_day = '%s/%s' % (self.model_log_dir, current_day)
        os.makedirs(self.current_day, exist_ok=True)

        index = get_log_dir_index(self.current_day)
        self.cfg_dir = '%s/%s' % (self.current_day, self.dataset + '_' + current_time + '_' + index)
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.result_dir = '%s/result' % self.cfg_dir

        os.makedirs(self.cfg_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, logger, patience=7, delta=0):
        """
        Args:
            logger: log the info to a .txt
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0
