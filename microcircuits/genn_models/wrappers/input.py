import numpy as np
from ..neurons import (rate_player_model)
from .neurons_bases import InputBase
# from microcircuits import Stages

__all__ = [
    "Input",
]


class Input(InputBase):

    def __init__(self, genn_model, n_neurons, rates, label=None,
                 params={}, init={}):

        layer = 0
        if label is None:
            label = "input_{:02d}_layer_{:05d}".format(
                        np.random.randint(0, 100), layer)

        super().__init__(
                    genn_model, n_neurons, rates, layer, label, params, init)

        # self._print_obj(self)

    def __repr__(self):
        return "{} type: {}".format(
            self.label, self.__class__.__name__)
