import numpy as np
import copy

class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.best_acc = -np.inf
        self.counter = 0
        self.best_model = None

    def step(self, val_acc, model):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_model = copy.deepcopy(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

    def should_stop(self):
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_model:
            model.__dict__.update(copy.deepcopy(self.best_model.__dict__))