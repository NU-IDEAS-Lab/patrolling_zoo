import os
import torch

class BaseAlgorithm:
    def __init__(self, env, device):
        self.env = env
        self.device = device

    def train(self, *args, seed=None, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, seed=None, **kwargs):
        raise NotImplementedError()
    
    def save(self, *args, path=None, **kwargs):
        ''' Saves the model. '''

        if path == None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            nameDir = os.path.dirname(os.path.realpath(__file__))
            nameBase = f"{self.__class__.__name__}-{timestamp}.pt"
            path = os.path.join(nameDir, "..", "models", nameBase)
        
        torch.save(
            {
                'model_state_dict': self.learner.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            path
        )