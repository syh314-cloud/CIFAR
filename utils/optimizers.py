import numpy as np
from .classic_optimizers import CLASSIC_OPTIMIZERS, get_optimizer as get_classic_optimizer
from .batch_optimizers import BATCH_OPTIMIZERS, get_optimizer as get_batch_optimizer

from .classic_optimizers import SGD, Momentum, Adam, Adagrad
from .batch_optimizers import BatchGD, OnlineGD, MiniBatchGD

class LearningRateScheduler:
    def __init__(self, initial_lr=0.01, decay_rate=0.1,patience=5):
        self.lr = initial_lr
        self.decay_rate = decay_rate
        self.patience = patience
        self.best_val_acc = 0
        self.wait = 0

    def step(self, val_acc):
        if val_acc > self.best_val_acc + 1e-4:
            self.best_val_acc = val_acc
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.lr = max(self.lr*self.decay_rate, 1e-5)
            self.wait = 0
        return self.lr

OPTIMIZER_GROUPS = {
    'classic': {
        'name': '经典优化器组',
        'optimizers': ['SGD', 'Momentum', 'Adam', 'Adagrad'],
        'description': '基于梯度的经典优化算法',
        'get_optimizer': get_classic_optimizer
    },
    'batch_strategy': {
        'name': '批次策略优化器组', 
        'optimizers': ['BatchGD', 'OnlineGD', 'MiniBatchGD'],
        'description': '不同批次处理策略的优化算法',
        'get_optimizer': get_batch_optimizer
    }
}

def get_optimizer(name, group='auto', **kwargs):
    if group == 'auto':
        if name in CLASSIC_OPTIMIZERS:
            return get_classic_optimizer(name, **kwargs)
        elif name in BATCH_OPTIMIZERS:
            return get_batch_optimizer(name, **kwargs)
        else:
            raise ValueError(f"未知优化器: {name}. 可用选项: {list(CLASSIC_OPTIMIZERS.keys()) + list(BATCH_OPTIMIZERS.keys())}")
    elif group == 'classic':
        return get_classic_optimizer(name, **kwargs)
    elif group == 'batch_strategy':
        return get_batch_optimizer(name, **kwargs)
    else:
        raise ValueError(f"未知优化器组: {group}")

def get_all_optimizers():
    all_optimizers = {}
    for name, config in CLASSIC_OPTIMIZERS.items():
        all_optimizers[name] = {
            'group': 'classic',
            'params': config['params'],
            'class': config['class']
        }
    for name, config in BATCH_OPTIMIZERS.items():
        all_optimizers[name] = {
            'group': 'batch_strategy',
            'params': config['params'],
            'class': config['class']
        }
    
    return all_optimizers

EXPERIMENT_CONFIGS = {
    'classic_comparison': {
        'optimizers': [
            ('SGD', {'lr': 0.01}),
            ('Momentum', {'lr': 0.01, 'momentum': 0.9}),
            ('Adam', {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}),
            ('Adagrad', {'lr': 0.01})
        ],
        'batch_size': 64,
        'description': '经典优化器对比实验'
    },
    'batch_strategy_comparison': {
        'optimizers': [
            ('BatchGD', {'lr': 0.01}),
            ('OnlineGD', {'lr': 0.001}),
            ('MiniBatchGD', {'lr': 0.01})
        ],
        'batch_sizes': {'BatchGD': 'all', 'OnlineGD': 1, 'MiniBatchGD': 64},
        'description': '批次策略优化器对比实验'
    }
}

