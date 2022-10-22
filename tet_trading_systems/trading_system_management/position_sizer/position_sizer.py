from abc import ABC, abstractmethod
from typing import Dict


class IPositionSizer(ABC):

    @property
    @abstractmethod
    def position_size_metric_str(self):
        raise NotImplementedError("Should contain a 'position_size_metric_str' property.")
    
    @abstractmethod
    def __call__(self) -> Dict:
        raise NotImplementedError('Should implement __call__()')
