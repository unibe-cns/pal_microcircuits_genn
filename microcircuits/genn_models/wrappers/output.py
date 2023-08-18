import numpy as np
from microcircuits.genn_models.neurons.output import (
    out_up_postsyn_model, out_target_postsyn_model,
    output_act_soma_params, output_act_basal_params,
    create_output_with_activation)
from microcircuits.genn_models.wrappers.neurons_bases import (
    PyramidalBase, SupervisorBase, OutputBase, BiasBase)
from microcircuits.genn_models.defaults import default_activation as def_act

__all__ = [
    "Output",
]


class Output(OutputBase):
    __post_syn = dict(
        from_higher=out_target_postsyn_model,
        from_lower=out_up_postsyn_model,
        from_bias=out_up_postsyn_model,
    )

    __default_params = {
        'g_leak': 0.1,
        'g_basal': 1.0,
        'g_target': 1.0,
        'threshold': 15.,
        'tau_highpass': 1.,
        'le_active': 1,
    }

    __default_global = {
        'learning_on': False
    }

    __default_init = {
        "V": 0.,
        "I": 0.,
        "v_target": 0.,
        "v_basal": 0.,
        "v_steady": 0.,
        "v_dot": 0.,
        "v_brev": 0.,
        "RefracTime": 1.,
        "dv": 0.,
        "rate": 0.,
        "rate_last": 0.,
        "rate_basal": 0.,
        "rate_highpass": 0.,
    }

    def __init__(self, genn_model, n_neurons, layer, label=None,
                 params={}, ini={}, global_params={}, activation=def_act):
        for k in self.__default_params:
            params[k] = params.get(k, self.__default_params[k])

        for k in self.__default_init:
            ini[k] = ini.get(k, self.__default_init[k])

        for k in self.__default_global:
            global_params[k] = global_params.get(k, self.__default_global[k])

        if label is None:
            "output_{:02d}_layer_{:05d}".format(
                np.random.randint(0, 100), layer)

        act_soma = activation(*output_act_soma_params)
        act_basal = activation(*output_act_basal_params)
        acts = {'soma': act_soma, 'basal': act_basal}
        neuron_model = create_output_with_activation(acts)

        obj = genn_model.add_neuron_population(
            label, n_neurons, neuron_model,
            params, ini
        )

        super().__init__(
                    n_neurons, layer, label, params, ini, obj, global_params)
        self.set_global_params()

        self.params['g_apical'] = 0.0

    def get_postsyn(self, source):
        if isinstance(source, PyramidalBase):
            return Output.__post_syn['from_lower']
        elif isinstance(source, SupervisorBase):
            return Output.__post_syn['from_higher']
        elif isinstance(source, BiasBase):
            return Output.__post_syn['from_bias']

        raise Exception("Invalid source type (or layer combination)")

    def __repr__(self):
        return "{} type: {}".format(
            self.label, self.__class__.__name__)

