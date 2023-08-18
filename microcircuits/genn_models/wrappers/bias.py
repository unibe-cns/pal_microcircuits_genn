from microcircuits.genn_models.wrappers.neurons_bases import BiasBase
from microcircuits.genn_models.neurons.bias import bias_model
from copy import deepcopy


class Bias(BiasBase):
    _default_params = {
        'value': 0.0
    }
    _default_init = {
        'rate': 0.0,
        'rate_last': 0.0,
    }

    def __init__(self, genn_model, value, label=None):
        self.params = deepcopy(Bias._default_params)
        self.params['value'] = value
        self.value = value
        self.init = deepcopy(Bias._default_init)
        self.init['rate'] = value
        self.init['rate_last'] = value

        if label is None:
            label = 'bias_neuron'

        obj = genn_model.add_neuron_population(
            label, 1, bias_model,
            self.params, self.init
        )

        # first param is number of neurons
        # second is layer numeric id
        super().__init__(
                    1, 0, label, self.params, self.init, obj)

        self.pop = self.obj

    def __repr__(self):
        return "{} type: {}".format(
            self.label, self.__class__.__name__)





