class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss  
        if self.best_score is None:
            self.best_score = score
            # Salva apenas o state_dict do melhor modelo
            self.best_model_wts = model.state_dict().copy()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict().copy()
            self.counter = 0

    def get_best_model(self, model):
        """
        Recebe uma instância de modelo e carrega o 
        state dict salvo durante o melhor estado.
        """
        model.load_state_dict(self.best_model_wts)
        return model
