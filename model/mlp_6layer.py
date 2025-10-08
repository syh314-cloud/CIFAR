try:
    import cupy as np
except ImportError:
    import numpy as np
from model.layers import Linear,ReLU,Softmax,Dropout

class MLP:
    def __init__(self,input_dim,hidden1,hidden2,hidden3,hidden4,hidden5,output_dim):
        self.linear1 = Linear(input_dim,hidden1)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(p=0.2)
        self.linear2 = Linear(hidden1,hidden2)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(p=0.3)
        self.linear3 = Linear(hidden2,hidden3)
        self.relu3 = ReLU()
        self.dropout3 = Dropout(p=0.3)
        self.linear4 = Linear(hidden3,hidden4)
        self.relu4 = ReLU()
        self.dropout4 = Dropout(p=0.3)
        self.linear5 = Linear(hidden4,hidden5)
        self.relu5 = ReLU()
        self.dropout5 = Dropout(p=0.3)
        self.linear6 = Linear(hidden5,output_dim)
        self.softmax = Softmax()
        self.layers = [self.linear1,self.linear2,self.linear3,self.linear4,self.linear5,self.linear6]

    def forward(self,x,training=True):
        z1 = self.linear1.forward(x)
        a1 = self.relu1.forward(z1)
        a1 = self.dropout1.forward(a1,training=training)
        z2 = self.linear2.forward(a1)
        a2 = self.relu2.forward(z2)
        a2 = self.dropout2.forward(a2,training=training)
        z3 = self.linear3.forward(a2)
        a3 = self.relu3.forward(z3)
        a3 = self.dropout3.forward(a3,training=training)
        z4 = self.linear4.forward(a3)
        a4 = self.relu4.forward(z4)
        a4 = self.dropout4.forward(a4,training=training)
        z5 = self.linear5.forward(a4)
        a5 = self.relu5.forward(z5)
        a5 = self.dropout5.forward(a5,training=training)
        z6 = self.linear6.forward(a5)
        y_pred = self.softmax.forward(z6)
        return y_pred

    def backward(self,grad_output):
        grad_z6 = self.softmax.backward(grad_output)
        grad_a5 = self.linear6.backward(grad_z6)
        grad_a5 = self.dropout5.backward(grad_a5)
        grad_z5 = self.relu5.backward(grad_a5)
        grad_a4 = self.linear5.backward(grad_z5)
        grad_a4 = self.dropout4.backward(grad_a4)
        grad_z4 = self.relu4.backward(grad_a4)
        grad_a3 = self.linear4.backward(grad_z4)
        grad_a3 = self.dropout3.backward(grad_a3)
        grad_z3 = self.relu3.backward(grad_a3)
        grad_a2 = self.linear3.backward(grad_z3)
        grad_a2 = self.dropout2.backward(grad_a2)
        grad_z2 = self.relu2.backward(grad_a2)
        grad_a1 = self.linear2.backward(grad_z2)
        grad_a1 = self.dropout1.backward(grad_a1)
        grad_z1 = self.relu1.backward(grad_a1)
        grad_input = self.linear1.backward(grad_z1)
        return grad_input

    def zero_grad(self):
        for layer in self.layers:
            layer.dw = np.zeros_like(layer.w)
            layer.db = np.zeros_like(layer.b)
            
