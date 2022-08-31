import random
import numpy as np
import torch

from ote.core.task import ITask
from ote.logger import get_logger

logger = get_logger()

class TorchTask(ITask):
    def __init__(self, spec, **kwargs):
        super().__init__(spec)

    @staticmethod
    def _set_random_seed(seed, deterministic=False):
        """Set random seed.

        Args:
            seed (int): Seed to be used.
            deterministic (bool): Whether to set the deterministic option for
                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
                to True and `torch.backends.cudnn.benchmark` to False.
                Default: False.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'Training seed was set to {seed} w/ deterministic={deterministic}.')
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def is_gpu_available():
        return torch.cuda.is_available()

    @staticmethod
    def get_current_device():
        return torch.cuda.current_device()

    @staticmethod
    def scatter_kwargs(inputs, kwargs):
        """Scatter with support for kwargs dictionary"""
        inputs = scatter_cpu(inputs) if inputs else []
        kwargs = scatter_cpu(kwargs) if kwargs else []
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)
        return inputs, kwargs