from abc import ABCMeta, abstractmethod
from typing import Tuple


class HandlerInterface(metaclass=ABCMeta):
    """Implements the interface that every handler used for prediction
    needs to implement.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "predict")
            and callable(subclass.predict)
            or NotImplemented
        )

    @abstractmethod
    def predict(self, **kwargs) -> Tuple[dict, dict, dict, dict, int]:
        """Prediction method that has to be implemented in the child classes.

        returns:
            pred_dict: (dict), the dictionary which then directly can be added to the prediction
                dict in the main,
            log_hist: (dict), the dict of history logs,
            log_obj: (dict), the dict of object logs,
            log_boundaries: (dict), the log of boundary logs,
            prediction_id: (int), The next prediction ID.
        """
        raise NotImplementedError
