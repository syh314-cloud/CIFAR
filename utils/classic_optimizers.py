try:
    import cupy as np
except ImportError:
    import numpy as np

class SGD:
    def step(self,lr,layers):
        for layer in layers:
            layer.w -= lr * layer.dw
            layer.b -= lr * layer.db

class Momentum:
    def __init__(self,lr,momentum):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def step(self,model):
        for i,layer in enumerate(model.layers):
            if f'w_{i}' not in self.velocities:
                self.velocities[f'w_{i}'] = np.zeros_like(layer.w)
                self.velocities[f'b_{i}'] = np.zeros_like(layer.b)
            self.velocities[f'w_{i}'] = self.momentum * self.velocities[f'w_{i}'] - self.lr * layer.dw
            self.velocities[f'b_{i}'] = self.momentum * self.velocities[f'b_{i}'] - self.lr * layer.db
            layer.w += self.velocities[f'w_{i}']
            layer.b += self.velocities[f'b_{i}']

class Adam:
    def __init__(self,lr,beta1,beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self,model):
        self.t += 1  
        self.epsilon = 1e-8
        
        for i,layer in enumerate(model.layers):
            if f'w_{i}' not in self.m:
                self.m[f'w_{i}'] = np.zeros_like(layer.w)
                self.m[f'b_{i}'] = np.zeros_like(layer.b)
                self.v[f'w_{i}'] = np.zeros_like(layer.w)
                self.v[f'b_{i}'] = np.zeros_like(layer.b)

            self.m[f'w_{i}'] = self.beta1 * self.m[f'w_{i}'] + (1-self.beta1)*layer.dw
            self.m[f'b_{i}'] = self.beta1 * self.m[f'b_{i}'] + (1-self.beta1)*layer.db
            self.v[f'w_{i}'] = self.beta2 * self.v[f'w_{i}'] + (1-self.beta2)*(layer.dw**2)
            self.v[f'b_{i}'] = self.beta2 * self.v[f'b_{i}'] + (1-self.beta2)*(layer.db**2)

            m_w_corrected = self.m[f'w_{i}'] / (1-self.beta1**self.t)
            m_b_corrected = self.m[f'b_{i}'] / (1-self.beta1**self.t)
            v_w_corrected = self.v[f'w_{i}'] / (1-self.beta2**self.t)
            v_b_corrected = self.v[f'b_{i}'] / (1-self.beta2**self.t)

            layer.w -= self.lr * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            layer.b -= self.lr * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)

class Adagrad:
    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.accumulated_squared_gradients = {}
    
    def step(self, model):
        for i, layer in enumerate(model.layers):
            if f'w_{i}' not in self.accumulated_squared_gradients:
                self.accumulated_squared_gradients[f'w_{i}'] = np.zeros_like(layer.w)
                self.accumulated_squared_gradients[f'b_{i}'] = np.zeros_like(layer.b)
            
            self.accumulated_squared_gradients[f'w_{i}'] += layer.dw ** 2
            self.accumulated_squared_gradients[f'b_{i}'] += layer.db ** 2
            
            adaptive_lr_w = self.lr / (np.sqrt(self.accumulated_squared_gradients[f'w_{i}']) + self.epsilon)
            adaptive_lr_b = self.lr / (np.sqrt(self.accumulated_squared_gradients[f'b_{i}']) + self.epsilon)
            
            layer.w -= adaptive_lr_w * layer.dw
            layer.b -= adaptive_lr_b * layer.db

CLASSIC_OPTIMIZERS = {
    'SGD': {'class': SGD, 'params': {'lr': 0.01}},
    'Momentum': {'class': Momentum, 'params': {'lr': 0.01, 'momentum': 0.9}},
    'Adam': {'class': Adam, 'params': {'lr': 0.001, 'beta1': 0.9, 'beta2': 0.999}},
    'Adagrad': {'class': Adagrad, 'params': {'lr': 0.01, 'epsilon': 1e-8}},
}

def get_optimizer(name, **kwargs):
    optimizer_info = CLASSIC_OPTIMIZERS[name]
    params = optimizer_info['params'].copy()
    params.update(kwargs) 
    return optimizer_info['class'](**params)
