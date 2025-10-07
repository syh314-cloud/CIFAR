try:
    import cupy as np
except ImportError:
    import numpy as np
from model.layers import Linear,ReLU,Softmax

class MLP:
    def __init__(self,input_dim,hidden_dim,output_dim):
        self.linear1 = Linear(input_dim,hidden_dim)
        self.relu = ReLU()
        self.linear2 = Linear(hidden_dim,output_dim)
        self.softmax = Softmax()
        self.layers = [self.linear1,self.linear2]

    def forward(self,x):
        z1 = self.linear1.forward(x)
        a1 = self.relu.forward(z1)
        z2 = self.linear2.forward(a1)
        y_pred = self.softmax.forward(z2)
        return y_pred

    def backward(self,grad_output):
        grad_z2 = self.softmax.backward(grad_output)  
        grad_a1 = self.linear2.backward(grad_z2)      
        grad_z1 = self.relu.backward(grad_a1)         
        grad_input = self.linear1.backward(grad_z1)   
        return grad_input

    def update(self,lr):
        for layer in self.layers:
            layer.w -= lr * layer.dw
            layer.b -= lr * layer.db

    def zero_grad(self):
        for layer in self.layers:
            layer.dw = np.zeros_like(layer.w)
            layer.db = np.zeros_like(layer.b)
            
