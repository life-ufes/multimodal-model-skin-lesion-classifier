import torch
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Attributes:
        patience (int): How many epochs to wait without improvement before stopping.
        delta (float): Minimum change in the monitored quantity to qualify as improvement.
        verbose (bool): Whether to print messages about improvements and the counter.
        path (str): Where to save the model if save_to_disk=True.
        save_to_disk (bool): If True, saves a new 'best' model (state_dict) to disk whenever val_loss improves.
    """

    def __init__(self, 
                 patience=7, 
                 delta=0.0, 
                 verbose=False,
                 path='checkpoint.pt',
                 save_to_disk=False):
        """
        Args:
            patience (int): Number of epochs to wait without improvement. Default: 7
            delta (float): Minimum improvement threshold. Default: 0.0
            verbose (bool): Print messages about early stopping progress. Default: False
            path (str): Path to save the best model state_dict (if save_to_disk=True). Default: 'checkpoint.pt'
            save_to_disk (bool): Whether to save the best model to disk. Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.save_to_disk = save_to_disk

        # Internal counters and tracking
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        """
        Checks if validation loss has improved enough (by self.delta). If not, increments counter. 
        If counter >= patience, sets self.early_stop to True.

        Args:
            val_loss (float): The current epoch's validation loss.
            model (torch.nn.Module): The model being trained.
        """
        # We use score = -val_loss because we want to *maximize* the negative of the loss
        # (i.e., minimize val_loss).
        score = -val_loss

        if self.best_score is None:
            # First epoch or first call
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_model_wts = model.state_dict()
            self._save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            # No improvement (or not enough improvement)
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} "
                      f"(val_loss: {val_loss:.6f} vs. best: {self.val_loss_min:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            # Improvement detected
            self.best_score = score
            self.val_loss_min = val_loss
            self.best_model_wts = model.state_dict()
            self.counter = 0
            self._save_checkpoint(val_loss, model)

    def _save_checkpoint(self, val_loss, model):
        """
        Saves the best model checkpoint if save_to_disk=True. Also prints 
        a message if verbose=True.
        """
        if self.save_to_disk:
            torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Validation loss decreased to {val_loss:.6f}. "
                  f"Saving best model...")

    def load_best_weights(self, model):
        """
        Loads the best weights found into the provided model instance.

        Args:
            model (torch.nn.Module): The model to load the best weights into.
        """
        model.load_state_dict(self.best_model_wts)
        return model
