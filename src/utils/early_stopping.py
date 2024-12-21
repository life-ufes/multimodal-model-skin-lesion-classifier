import torch
import torch.nn as nn
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # Número de épocas sem melhoria antes de parar
        self.delta = delta  # O mínimo de melhoria exigido
        self.counter = 0  # Contador de épocas sem melhoria
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss  # Negativo da perda, porque queremos minimizar a perda
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

    def get_best_model(self):
        return self.best_model_wts