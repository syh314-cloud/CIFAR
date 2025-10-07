from utils.data_loader import train_images,train_labels,val_images,val_labels,test_images,test_labels
from utils.early_stopping import EarlyStopping
from model.baseline import SoftmaxClassifier
from model.mlp_2layer import MLP
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.classic_optimizers import SGD, Momentum, Adam
from utils.optimizers import LearningRateScheduler
from utils.data_augmentation import augment_images
from model.layers import DropoutScheduler
from utils.loss import L2Scheduler
import numpy as np

SEED = 2023
np.random.seed(SEED)


#model = SoftmaxClassifier(train_images.shape[1],train_labels.shape[1])
#model = MLP(train_images.shape[1],512,train_labels.shape[1])
model = MLP(train_images.shape[1],1024,512,256,train_labels.shape[1])
loss_fn = CrossEntropyLoss()
l2_scheduler = L2Scheduler(base_lambda=1e-4)
lr = 0.001

#optimizer = Momentum(lr, momentum=0.9)  
optimizer = Adam(lr, beta1=0.9, beta2=0.999)
#scheduler = LearningRateScheduler(initial_lr=lr, patience=5, decay_rate=0.5)  
dropout_scheduler = DropoutScheduler(base_p=0.3, min_p=0.1, max_p=0.6, patience=3, factor=0.05)
batch_size = 64
epochs = 100
#early_stopper = EarlyStopping(patience=10)
lambda_l2 = l2_scheduler.base_lambda

for epoch in range(epochs):
    if epoch < 20:
        model.dropout1.p = 0.0
        model.dropout2.p = 0.0
        model.dropout3.p = 0.0
    else:
        model.dropout1.p = 0.2
        model.dropout2.p = 0.3
        model.dropout3.p = 0.3
    np.random.seed(SEED + epoch)
    idx = np.random.permutation(train_images.shape[0])
    shuffled_images = train_images[idx]
    shuffled_labels = train_labels[idx]
    for i in range(0,shuffled_images.shape[0],batch_size):
        x = shuffled_images[i:i+batch_size]
        y = shuffled_labels[i:i+batch_size]
        x = x.reshape(-1, 3, 32, 32)
        x = augment_images(x, seed=SEED + epoch * 1000 + i)
        x = x.reshape(x.shape[0], -1)
        model.zero_grad()
        y_pred = model.forward(x,training=True)
        #y_pred = model.forward(x)
        #loss = model.compute_loss(y_pred,y)
        #model.backward(x,y)
        #model.update(lr)
        loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
        grad_output = loss_fn.backward()
        model.backward(grad_output)
        
        for layer in model.layers:
            if hasattr(layer, 'w'):
                layer.dw += lambda_l2 * layer.w
        
        optimizer.step(model) 
        #print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, Loss {loss}")
    val_pred = model.forward(val_images, training=False)  
    #val_pred = model.forward(val_images)
    val_acc = np.mean(np.argmax(val_pred,axis=1) == np.argmax(val_labels,axis=1))
    #new_lr = scheduler.step(val_acc)
    #optimizer.lr = new_lr
    """
    new_p = dropout_scheduler.step(val_acc)
    model.dropout1.p = new_p
    model.dropout2.p = new_p
    model.dropout3.p = new_p
    """
    print(f"Epoch {epoch+1}, Val Accuracy {val_acc:.4f}")
    #lambda_l2 = l2_scheduler.step(val_acc)
    """
    early_stopper.step(val_acc, model)
    if early_stopper.should_stop():
        early_stopper.restore_best(model)
        print(f"Early stopping at epoch {epoch+1}, restored best model.")
        break
    """

test_pred = model.forward(test_images,training=False)
#test_pred = model.forward(test_images)
test_acc = np.mean(np.argmax(test_pred,axis=1) == np.argmax(test_labels,axis=1))
print(f"Test Accuracy:{test_acc}")
