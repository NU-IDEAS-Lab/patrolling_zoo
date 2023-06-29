class BaseAlgorithm:
    def __init__(self, env, device):
        self.env = env
        self.device = device

    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()