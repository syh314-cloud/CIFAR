try:
    import cupy as np
except ImportError:
    import numpy as np
from model.layers import Softmax

class CrossEntropyLoss:
    def forward(self,y_pred,y_true, model=None, lambda_l2=1e-4):
        self.y_pred = y_pred
        self.y_true = y_true
        eps = 1e-7
        loss = -np.sum(y_true * np.log(y_pred+eps))/y_pred.shape[0]
        if model is not None:
            l2_reg = 0
            for layer in model.layers:
                if hasattr(layer, 'w'):
                    l2_reg += np.sum(layer.w ** 2)
            loss += (lambda_l2 / (2 * y_true.shape[0])) * l2_reg
        return loss

    def backward(self):
        dz = (self.y_pred - self.y_true)
        return dz

class LabelSmoothingLoss:
    def __init__(self,alpha=0.1):
        self.alpha = alpha

    def forward(self,y_pred,y_true, model=None, lambda_l2=1e-4):
        self.y_pred = y_pred
        self.y_true = y_true
        K = y_true.shape[1]
        self.smooth_labels = (1-self.alpha)*y_true + self.alpha/K  # 保存为实例变量
        eps = 1e-7
        loss = -np.sum(self.smooth_labels * np.log(y_pred+eps))/y_pred.shape[0]
        if model is not None:
            l2_reg = 0
            for layer in model.layers:
                if hasattr(layer, 'w'):
                    l2_reg += np.sum(layer.w ** 2)
            loss += (lambda_l2 / (2 * y_true.shape[0])) * l2_reg
        return loss

    def backward(self):
        # 使用平滑后的标签计算梯度，而不是原始标签
        dz = (self.y_pred - self.smooth_labels)
        return dz

class FocalLoss:
    def __init__(self,gamma=2.0):
        self.gamma = gamma
        self.eps = 1e-7
    
    def forward(self,y_pred,y_true, model=None, lambda_l2=1e-4):
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        p_t = np.sum(y_true * y_pred,axis=1)
        loss = -np.mean((1 - p_t)**self.gamma * np.log(p_t))
        if model is not None:
            l2_reg = 0
            for layer in model.layers:
                if hasattr(layer, 'w'):
                    l2_reg += np.sum(layer.w ** 2)
            loss += (lambda_l2 / (2 * y_true.shape[0])) * l2_reg
        return loss
    
    def backward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        p_t = np.sum(y_true * y_pred, axis=1, keepdims=True)
        log_p_t = np.log(p_t)
        grad = -y_true * ((1 - p_t) ** self.gamma * (1 + self.gamma * p_t * log_p_t)) / p_t
        grad /= y_true.shape[0]
        return grad

class L2Scheduler:
    def __init__(self, base_lambda=1e-4, min_lambda=1e-6, max_lambda=1e-2, patience=3, factor=2):
        self.base_lambda = base_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.patience = patience
        self.factor = factor
        self.best_acc = 0
        self.wait = 0

    def step(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.wait = 0
            self.base_lambda = max(self.min_lambda, self.base_lambda / self.factor)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.base_lambda = min(self.max_lambda, self.base_lambda * self.factor)
                self.wait = 0
        return self.base_lambda