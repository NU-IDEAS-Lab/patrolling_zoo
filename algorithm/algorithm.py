class BaseAlgorithm:
    def __init__(self, env, device):
        self.env = env
        self.device = device

    def train(self, *args, seed=None, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, seed=None, **kwargs):
        raise NotImplementedError()