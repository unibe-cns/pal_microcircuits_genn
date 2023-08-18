import numpy as np
from ..neurons import (rate_player_model)
from .neurons_bases import SupervisorBase
from pprint import pprint

__all__ = [
    "Supervisor",
]


class Supervisor(SupervisorBase):
    __default_params = {
        "period": 100,
        "tau": 3,  # use 0 to remove low-pass filtering
        "mute_t": 0,
        "noise_amp": 0,
        "base_rate": 0,
    }

    __default_init = {
        "last_t": 0.0,
        "rate": 0.,
        "rate0": 0.,
        "rate_last": 0.,
        "startStim": None,
        "endStim": None,
        "index": 0,
    }

    def __init__(self, genn_model, n_neurons, rates, label=None,
                 params={}, init={}, layer=3):
        if label is None:
            "supervisor_{:02d}_layer_{:05d}".format(
                np.random.randint(0, 100), layer)

        super().__init__(
                    genn_model, n_neurons, rates, layer, label, params, init)


    def __repr__(self):
        return "{} type: {}".format(
            self.label, self.__class__.__name__)
