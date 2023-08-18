import numpy as np
from microcircuits.genn_models.neurons.interneuron import (
    inter_down_postsyn_model, inter_pyramidal_postsyn_model,
    interneuron_act_soma_args, interneuron_act_dendrite_args,
    create_interneuron_with_activation)
from microcircuits.genn_models.wrappers.neurons_bases import (
    PyramidalBase, OutputBase, InterBase, BiasBase)
from microcircuits.genn_models.defaults import default_activation as def_act

__all__ = [
    "Inter",
]


class Inter(InterBase):
    __post_syn = dict(
        from_higher=inter_down_postsyn_model,
        from_pyramidal=inter_pyramidal_postsyn_model,
        from_bias=inter_pyramidal_postsyn_model,
    )

    __default_params = {
        'g_leak': 0.1,
        'g_soma': 0.8,
        'g_dendrite': 1.0,
        'threshold': 15.,
        'noise_amplitude': 0.0,
        'le_active': 1,
    }

    __default_init = {
        "V": 0.,
        "I": 0.,
        "v_down": 0.,
        "v_dendrite": 0.,
        "v_steady": 0.,
        "v_dot": 0.,
        "v_brev": 0.,
        "RefracTime": 1.,
        "dv": 0.,
        "rate": 0.,
        "rate_last": 0.,
        "rate_dendrite": 0.,
        "rate_dendrite_last": 0.,
    }

    def __init__(self, genn_model, n_neurons, layer, label=None,
                 params={}, init={}, activation=def_act):
        for k in self.__default_params:
            params[k] = params.get(k, self.__default_params[k])

        for k in self.__default_init:
            init[k] = init.get(k, self.__default_init[k])

        if label is None:
            "inter_{:02d}_layer_{:05d}".format(
                np.random.randint(0, 100), layer)

        act_soma = activation(*interneuron_act_soma_args)
        act_dendrite = activation(*interneuron_act_dendrite_args)
        acts = {'soma': act_soma, 'dendrite': act_dendrite}
        neuron_model = create_interneuron_with_activation(acts)

        obj = genn_model.add_neuron_population(
                    label, n_neurons, neuron_model,
                    params, init)

        super().__init__(n_neurons, layer, label, params, init, obj)

        # self._print_obj(self)

    def get_postsyn(self, source):
        if isinstance(source, PyramidalBase) and source.layer == self.layer:
            return Inter.__post_syn['from_pyramidal']
        elif isinstance(source, BiasBase):
            return Inter.__post_syn['from_bias']
        elif isinstance(source, OutputBase):
            return Inter.__post_syn['from_higher']
        elif isinstance(source, PyramidalBase) and source.layer > self.layer:
            return Inter.__post_syn['from_higher']

        raise Exception("Invalid source type (or layer combination)")

    def __repr__(self):
        return "{} type: {}".format(
            self.label, self.__class__.__name__)


