try:
    import cupy as np
except ImportError:
    import numpy as np

class SoftmaxClassifier:
    def __init__(self,input_dim,num_classes):
        #输入图片维度 N*D,45000*3072
        w = np.random.randn(input_dim,num_classes)
        b = np.random.randn(1,num_classes)
        self.w = w
        self.b = b
    
    def forward(self,x):
        z = np.dot(x,self.w) + self.b # N*num_classes
        # Softmax激活函数（数值稳定版本）
        z_stable = z - np.max(z,axis=1,keepdims=True)
        z_exp = np.exp(z_stable)
        softmax_output = z_exp/np.sum(z_exp,axis=1,keepdims=True)  
        return softmax_output

    def compute_loss(self,y_pred,y_true):
        eps = 1e-7
        y_pred = np.clip(y_pred,eps,1-eps) #防止log0
        loss = -np.sum(y_true * np.log(y_pred))/y_pred.shape[0] #交叉熵损失
        return loss

    def backward(self,x,y):
        y_pred = self.forward(x)
        dz = (y_pred - y)/y_pred.shape[0]
        dw = np.dot(x.T,dz)
        db = np.sum(dz,axis=0,keepdims=True)
        self.dw = dw
        self.db = db

    def update(self,lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db


