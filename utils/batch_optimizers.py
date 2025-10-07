try:
    import cupy as np
except ImportError:
    import numpy as np

class BatchGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.accumulated_gradients = {}
        self.batch_count = 0
    
    def accumulate_gradients(self, model):
        self.batch_count += 1
        for i, layer in enumerate(model.layers):
            if f'w_{i}' not in self.accumulated_gradients:
                self.accumulated_gradients[f'w_{i}'] = np.zeros_like(layer.w)
                self.accumulated_gradients[f'b_{i}'] = np.zeros_like(layer.b)
                
            self.accumulated_gradients[f'w_{i}'] += layer.dw
            self.accumulated_gradients[f'b_{i}'] += layer.db
    
    def step(self, model):
        if self.batch_count == 0:
            return
        for i, layer in enumerate(model.layers):
            avg_dw = self.accumulated_gradients[f'w_{i}'] / self.batch_count
            avg_db = self.accumulated_gradients[f'b_{i}'] / self.batch_count
            layer.w -= self.lr * avg_dw
            layer.b -= self.lr * avg_db
        self.reset()
    
    def reset(self):
        self.accumulated_gradients = {}
        self.batch_count = 0

class OnlineGD:
    def __init__(self, lr=0.001):  
        self.lr = lr
    
    def step(self, model):
        for layer in model.layers:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db

class MiniBatchGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, model):
        for layer in model.layers:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db

class AdaptiveBatchGD:
    def __init__(self, lr=0.01, min_batch_size=16, max_batch_size=256):
        self.lr = lr
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gradient_history = []
        self.current_batch_size = 64
    
    def step(self, model):
        grad_norm = 0
        for layer in model.layers:
            grad_norm += np.sum(layer.dw**2) + np.sum(layer.db**2)
        grad_norm = np.sqrt(grad_norm)
            
        for layer in model.layers:
            layer.w -= self.lr * layer.dw
            layer.b -= self.lr * layer.db
        
        self.gradient_history.append(grad_norm)
        if len(self.gradient_history) > 10:
            self.gradient_history.pop(0)
    
    def get_adaptive_batch_size(self):
        if len(self.gradient_history) < 5:
            return self.current_batch_size
        grad_var = np.var(self.gradient_history[-5:])
        if grad_var > 0.1:  
            self.current_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
        elif grad_var < 0.01: 
            self.current_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
        return self.current_batch_size

BATCH_OPTIMIZERS = {
    'BatchGD': {
        'class': BatchGD, 
        'params': {'lr': 0.01},
    },
    'OnlineGD': {
        'class': OnlineGD, 
        'params': {'lr': 0.001},
    },
    'MiniBatchGD': {
        'class': MiniBatchGD, 
        'params': {'lr': 0.01},
    },
    'AdaptiveBatchGD': {
        'class': AdaptiveBatchGD, 
        'params': {'lr': 0.01},
    },
}

def get_optimizer(name, **kwargs):
    optimizer_info = BATCH_OPTIMIZERS[name]
    params = optimizer_info['params'].copy()
    params.update(kwargs)  
    return optimizer_info['class'](**params)

