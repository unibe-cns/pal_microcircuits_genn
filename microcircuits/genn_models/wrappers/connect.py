import numpy as np
from pygenn import genn_wrapper
from pygenn.genn_model import init_connectivity

from ..synapses import (
    up_synapse_model, inter_down_synapse_model,
    output_down_synapse_model, pyramidal_down_synapse_model,
    i2p_synapse_model, p2i_synapse_model
)
from .neurons_bases import ConnectBase
from . import Input, Pyramidal, Inter, Supervisor, Output
from .bias import Bias


class Connector(ConnectBase):
    __default_params = {
    }
    __default_init = {
        "g": "uniform_symmetric",
    }
    __default_global = {
    }

    def __init__(self, genn_model, pre, post, params={}, init={},
                 global_params={}, conn_init=None, label=None):

        for k in self.__default_params:
            params[k] = params.get(k, self.__default_params[k])

        for k in self.__default_init:
            init[k] = init.get(k, self.__default_init[k])

        for k in self.__default_global:
            global_params[k] = global_params.get(k, self.__default_global[k])

        if label is None:
            label = "synapse__{}__TO__{}".format(pre.label, post.label)

        if isinstance(init['g'], np.ndarray) and len(init['g'].shape) > 1:
            init['g'] = init['g'].flatten()

        self.syn_model = self.__get_synapse_model(pre, post)
        self.matrix_type = self.__get_matrix_type(pre, post)
        self.post_syn = post.get_postsyn(pre)

        _conn_init = None
        if conn_init is not None:
            _init = conn_init['init']
            _params = conn_init['params']
            _conn_init = init_connectivity(_init, _params)

        self.conn_init_params = conn_init
        self.conn_init = _conn_init

        obj = genn_model.add_synapse_population(
            label, self.matrix_type, genn_wrapper.NO_DELAY,
            pre.obj, post.obj, self.syn_model,
            params, init,
            {}, {},
            self.post_syn, {}, {},
            connectivity_initialiser=_conn_init
        )

        super().__init__(pre, post, label, params, init, obj, global_params)

        # self._print_obj(self)
    # def __str__(self):
    #     return self.label
    #
    # def __repr__(self):
    #     return "{}\n{}".format(self.label, self._str_params())

    @staticmethod
    def is_one_to_one(pre, post):
        back = pre.layer == (post.layer + 1)

        if isinstance(pre, Supervisor) and isinstance(post, Output):
            return True

        if isinstance(pre, Output) and isinstance(post, Inter) and back:
            return True

        if isinstance(pre, Pyramidal) and isinstance(post, Inter) and back:
            return True

        return False

    @staticmethod
    def __get_matrix_type(pre, post):
        if Connector.is_one_to_one(pre, post):
            return "SPARSE_INDIVIDUALG"
        else:
            return "DENSE_INDIVIDUALG"

    # def __get_conn_type(self):

    @staticmethod
    def __get_synapse_model(pre, post):
        forward = (pre.layer + 1) == post.layer
        back = pre.layer == (post.layer + 1)
        lateral = pre.layer == post.layer

        # Bias
        if isinstance(pre, Bias) and isinstance(post, Pyramidal):
            return up_synapse_model

        elif isinstance(pre, Bias) and isinstance(post, Inter):
            return p2i_synapse_model

        elif isinstance(pre, Bias) and isinstance(post, Output):
            return up_synapse_model

        # Feed-forward
        elif isinstance(pre, Input) and isinstance(post, Pyramidal) and forward:
            return up_synapse_model

        elif isinstance(pre, Pyramidal) and isinstance(post, Output) and forward:
            return up_synapse_model

        elif isinstance(pre, Pyramidal) and isinstance(post, Pyramidal) and forward:
            return up_synapse_model

        # Inter-neuron
        elif isinstance(pre, Pyramidal) and isinstance(post, Inter) and lateral:
            return p2i_synapse_model

        elif isinstance(pre, Inter) and isinstance(post, Pyramidal) and lateral:
            return i2p_synapse_model

        # Feed-back
        elif isinstance(pre, Supervisor) and isinstance(post, Output):
            return output_down_synapse_model

        elif isinstance(pre, Output) and isinstance(post, Pyramidal) and back:
            return pyramidal_down_synapse_model

        elif isinstance(pre, Pyramidal) and isinstance(post, Pyramidal) and back:
            return pyramidal_down_synapse_model

        elif isinstance(pre, Output) and isinstance(post, Inter) and back:
            return inter_down_synapse_model

        elif isinstance(pre, Pyramidal) and isinstance(post, Inter) and back:
            return inter_down_synapse_model

        raise Exception("Not supported Connection type / Synapse.\n"
                        "{} to {}".format(pre, post))



