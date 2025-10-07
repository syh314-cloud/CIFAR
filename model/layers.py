from re import X
try:
    import cupy as np
except ImportError:
    import numpy as np

class Linear:
    def __init__(self,input_dim,output_dim):
        w = np.random.randn(input_dim,output_dim)*np.sqrt(2.0/(input_dim+output_dim))
        b = np.zeros((1,output_dim))
        self.w = w
        self.b = b
        self.dw = None
        self.db = None

    def forward(self,x):
        self.x = x
        z = np.dot(x,self.w) + self.b
        return z
        
    def backward(self,grad_z):
        m = self.x.shape[0] 
        self.dw = np.dot(self.x.T,grad_z) / m
        self.db = np.sum(grad_z,axis=0,keepdims=True) / m
        dx = np.dot(grad_z,self.w.T)
        return dx
    
class ReLU:
    def forward(self,x):
        self.x = x
        y = np.maximum(0,x)
        return y

    def backward(self,grad_z):
        dx = grad_z * (self.x>0)
        return dx

class Dropout:
    def __init__(self,p):
        self.p = p
        self.mask = None

    def forward(self,x,training=True):
        if training:
            self.mask = np.random.rand(*x.shape) > self.p
            out = (x * self.mask)/(1-self.p)
            return out
        else:
            return x

    def backward(self,grad_output):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask/(1-self.p)

class Softmax:
    def forward(self,z):
        z_stable = z - np.max(z,axis=1,keepdims=True)
        z_exp = np.exp(z_stable)
        self.output = z_exp/np.sum(z_exp,axis=1,keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        return grad_output

    
class DropoutScheduler:
    def __init__(self, base_p=0.3, min_p=0.1, max_p=0.6, patience=3, factor=0.05):
        self.base_p = base_p
        self.min_p = min_p
        self.max_p = max_p
        self.factor = factor
        self.patience = patience
        self.best_acc = 0
        self.wait = 0

    def step(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.wait = 0
            self.base_p = max(self.min_p, self.base_p - self.factor)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.base_p = min(self.max_p, self.base_p + self.factor)
                self.wait = 0
        return self.base_p