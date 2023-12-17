import abc

class Function(abc.ABC):
    """Class of functions which maps d dimensional vectors to scalar"""
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def value(self, x):
        pass

    @abc.abstractmethod
    def grad(self, x):
        pass