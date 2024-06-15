""" Interface for standardizing structure of prediction models """
import abc

class ModelInterface(metaclass=abc.ABCMeta):
    """Interface to make sure all models contain the same functions."""
    @abc.abstractmethod
    def forward(self, X):
        """ Make predictions on the provided samples. """

    @abc.abstractmethod
    def train(self, X, y):
        """ Train the model on the provided samples. """

    @abc.abstractmethod
    def evaluate(self, X, y):
        """ Train the model on the provided samples. """

    @abc.abstractmethod
    def save(self):
        """ Save the model for later use. """
        
    @abc.abstractmethod
    def _get_name(self):
        """ Save the model for later use. """
        
