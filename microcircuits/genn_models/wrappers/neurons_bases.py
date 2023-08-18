import numpy as np
from . import PrinterMixin
from warnings import warn
from microcircuits.genn_models.common import Stages
from ..neurons import (rate_player_model)
from pprint import pprint

class NeuronsBase(PrinterMixin):
    def __init__(self, size, layer, label, params, init, obj, global_params=None):
        self.size = size
        self.layer = layer
        self.label = label
        self.params = params
        self.init = init
        self.obj = obj
        if hasattr(obj, "neuron"):
            self.name = self.obj.neuron.__class__.__name__
        else:
            self.name = self.obj.w_update.__class__.__name__

        self.global_params = global_params

        self.set_global_params()
        self.var_views = {}
        self.views = {}

    def set_global_params(self):
        if self.global_params is not None:
            for k in self.global_params:
                self.obj.set_extra_global_param(k, self.global_params[k])

    def setup_global_views(self):
        if self.global_params is not None:
            for k in self.global_params:
                self.views[k] = self.obj.extra_global_params[k].view

    def setup_variable_views(self):
        for k in self.obj.vars:
            self.var_views[k] = self.obj.vars[k].view

    def get_view(self, k):
        return self.views[k]

    def set_global(self, variable, value):
        if variable in self.views:
            if hasattr(value, 'shape'):
                n_vals = len(value)
                self.views[variable][:n_vals] = value
                self.obj.push_extra_global_param_to_device(variable)
            else:
                self.views[variable][:] = value
        else:
            raise Exception(
                "Global parameter '{}' not found in {}".format(
                    variable, self.label))

    def set_variable(self, variable, value):
        if variable in self.var_views:
            end = len(value) if hasattr(value, 'shape') else None
            self.var_views[variable][:end] = value
            self.obj.push_var_to_device(variable)
        else:
            raise Exception(
                "Variable '{}' not found in {}".format(variable, self.label))

    def _print_global_param(self, p):
        if p in self.views and p in self.obj.extra_global_params:
            print(self.label)
            print("\t{} = view {}\tdevice {}".format(
                p, self.views[p], self.obj.extra_global_params[p].view))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = """{}[{}]({}) at layer {}\n\t{}
        """.format(self.label, self.name, self.size, self.layer,
                   self._str_params(3))
        return s


class RatePlayerBase(NeuronsBase):
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
        "step_counter": 0,
    }

    def __init__(self, genn_model, size, rates, layer, label, params, init):

        for k in self.__default_params:
            params[k] = params.get(k, self.__default_params[k])

        for k in self.__default_init:
            init[k] = init.get(k, self.__default_init[k])

        for stage in rates:
            rt_shape = rates[stage].shape
            if size != rt_shape[0]:
                raise Exception("The first dimension of the "
                                "rates array should be equal to the number of "
                                "neurons in the population.")

        sorted_stage_names = sorted(rates.keys())
        self.rates = {stage: rates[stage].flatten() for stage in sorted_stage_names}
        self.accum_rates = np.hstack([self.rates[s]
                                      for s in sorted_stage_names])

        _max = Stages.TRAINING
        for stage in sorted_stage_names:
            if rates[_max].shape[1] <= rates[stage].shape[1]:
                _max = stage
        self.longest_stage = _max

        self.n_pat = {stage: len(self.rates[stage]) // size
                      for stage in sorted_stage_names}

        self.rates_counts = {stage: [len(n) for n in rates[stage]]
                             for stage in sorted_stage_names}

        # can't change size of arrays at runtime so we need to take the largest on for all stages
        self.rate_shuffler = {stage: np.arange(self.n_pat[self.longest_stage], dtype=np.int) for stage in sorted_stage_names}

        ends = {}
        starts = {}
        for stage_idx, stage in enumerate(sorted_stage_names):
            if stage_idx == 0:
                prev_end = 0
            else:
                prev_stage = sorted_stage_names[stage_idx - 1]
                prev_end = ends[prev_stage][-1]

            ends[stage] = np.cumsum(self.rates_counts[stage])
            ends[stage] += prev_end

            starts[stage] = np.empty_like(ends[stage])
            starts[stage][0] = prev_end
            starts[stage][1:] = ends[stage][0:-1]

        self.starts = starts
        self.ends = ends
        self.rate0 = {stage: [self.accum_rates[s] if params["mute_t"] == 0 else 0
                      for s in self.starts[stage]] for stage in self.starts}

        stage = self.longest_stage
        init['startStim'] = self.starts[stage]
        init['endStim'] = self.ends[stage]
        init['rate'] = self.rate0[stage]
        init['rate0'] = self.rate0[stage]
        init['rate_last'] = self.rate0[stage]

        obj = genn_model.add_neuron_population(
            label, size, rate_player_model,
            params, init
        )

        global_params = {'rates_list' : self.accum_rates,
                         'rate_shuffler': self.rate_shuffler[stage],}
        super().__init__(size, layer, label, params, init, obj, global_params)
        self.set_global_params()
        self.first_reset = False

    def reset_player(self, stage, shuffled_indices=None):
        self.set_variable('index', 0)
        self.set_variable('last_t', 0)
        self.set_variable('step_counter', 0)
        self.set_variable('startStim', self.starts[stage])
        self.set_variable('endStim', self.ends[stage])

        if shuffled_indices is None:
            start_idx = 0
        else:
            start_idx = shuffled_indices[0]
        r0 = self.accum_rates[self.starts[stage] + start_idx]
        self.set_variable('rate0', r0)
        self.set_variable('rate', r0)
        self.set_variable('rate_last', r0)
        if not (shuffled_indices is None):
            self.rate_shuffler[stage][:len(shuffled_indices)] = shuffled_indices
        self.set_global('rate_shuffler', self.rate_shuffler[stage])


class BiasBase(NeuronsBase):
    def __init__(self, size, layer, label, params, init, obj, global_params=None):
        super().__init__(size, layer, label, params, init, obj, global_params)


class InterBase(NeuronsBase):
    def __init__(self, size, layer, label, params, init, obj, global_params=None):
        super().__init__(size, layer, label, params, init, obj, global_params)


class OutputBase(NeuronsBase):
    def __init__(self, size, layer, label, params, init, obj, global_params=None):
        super().__init__(size, layer, label, params, init, obj, global_params)


class PyramidalBase(NeuronsBase):
    def __init__(self, size, layer, label, params, init, obj, global_params=None):
        super().__init__(size, layer, label, params, init, obj, global_params)


class SupervisorBase(RatePlayerBase):
    pass

class InputBase(RatePlayerBase):
    pass

class ConnectBase(NeuronsBase):
    def __init__(self, pre, post, label, params, init, obj, global_params=None):
        self.pre = pre
        self.post = post
        super().__init__(None, None, label, params, init, obj, global_params)

