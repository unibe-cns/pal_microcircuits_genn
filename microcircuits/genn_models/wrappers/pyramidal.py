import numpy as np
from microcircuits.genn_models.neurons.pyramidal import (
    pyr_up_postsyn_model, pyr_inter_postsyn_model,
    pyr_down_postsyn_model, pyramidal_act_soma_params,
    pyramidal_act_basal_params, create_pyramidal_with_activation)
from microcircuits.genn_models.wrappers.neurons_bases import (
    InterBase, OutputBase, InputBase, PyramidalBase, BiasBase)
from microcircuits.genn_models.defaults import default_activation as def_act

__all__ = [
    "Pyramidal",
]


class Pyramidal(PyramidalBase):
    __post_syn = dict(
        from_lower=pyr_up_postsyn_model,
        from_bias=pyr_up_postsyn_model,
        from_higher=pyr_down_postsyn_model,
        from_inter=pyr_inter_postsyn_model,
    )
    __default_params = {
        'g_leak': 0.1,
        'g_basal': 1.0,
        'g_apical': 0.8,
        'threshold': 15.,
        'sigma_noise': 0.0,
        'tau_noise': 1.,
        'tau_highpass': 1.,
        'le_active': 1,
    }

    __default_init = {
        "V": 0.,
        "I": 0.,
        "v_apical": 0.,
        "v_steady": 0.,
        "noise": 0.,
        "v_dot": 0.,
        "v_brev": 0.,
        "v_basal": 0.,
        "RefracTime": 1.,
        "dv": 0.,
        "rate": 0.,
        "rate_last": 0.,
        "rate_basal": 0.,
        "rate_highpass": 0.,
    }

    def __init__(self, genn_model, n_neurons, layer, label=None,
                 params={}, init={}, activation=def_act):
        for k in self.__default_params:
            params[k] = params.get(k, self.__default_params[k])

        for k in self.__default_init:
            init[k] = init.get(k, self.__default_init[k])

        if label is None:
            "pyramidal_{:02d}_layer_{:05d}".format(
                np.random.randint(0, 100), layer)

        act_soma = activation(*pyramidal_act_soma_params)
        act_basal = activation(*pyramidal_act_basal_params)
        acts = {'soma': act_soma, 'basal': act_basal}
        neuron_model = create_pyramidal_with_activation(acts)

        obj = genn_model.add_neuron_population(
            label, n_neurons, neuron_model,
            params, init
        )

        super().__init__(n_neurons, layer, label, params, init, obj)

        # self._print_obj(self)

    def get_postsyn(self, source):
        if isinstance(source, InputBase):
            return Pyramidal.__post_syn['from_lower']
        elif isinstance(source, BiasBase):
            return Pyramidal.__post_syn['from_bias']
        elif isinstance(source, InterBase):
            return Pyramidal.__post_syn['from_inter']
        elif isinstance(source, OutputBase):
            return Pyramidal.__post_syn['from_higher']
        elif isinstance(source, Pyramidal) and source.layer > self.layer:
            return Pyramidal.__post_syn['from_higher']
        elif isinstance(source, Pyramidal) and source.layer < self.layer:
            return Pyramidal.__post_syn['from_lower']

        raise Exception("Invalid source type (or layer combination)")

    def __repr__(self):
        return "{} type: {}".format(
            self.label, self.__class__.__name__)



