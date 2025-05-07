from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train(self):
        """학습 루프 진입점"""
        pass
