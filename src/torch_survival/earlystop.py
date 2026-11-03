import copy
from typing import Literal


class EarlyStopper:
    def __init__(self, mode: Literal['min', 'max'] = 'min', patience=10):
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.best_model_state = None  # To store weights in RAM

    def should_stop(self, model, score):
        if (self.mode == 'min' and score < self.best_score) or (self.mode == 'max' and score > self.best_score):
            self.best_score = score
            self.counter = 0
            # Deep copy the state dict to avoid reference issues
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
