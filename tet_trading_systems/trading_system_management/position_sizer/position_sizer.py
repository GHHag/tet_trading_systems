from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List

from TETrading.position.position import Position


class IPositionSizer(ABC):

    @property
    @abstractmethod
    def position_size_metric_str(self):
        raise NotImplementedError("Should contain a 'position_size_metric_str' property.")
    
    @abstractmethod
    def __call__(self, position_list: List[Position], period_len: int) -> Dict:
        raise NotImplementedError('Should implement __call__()')
