from .input import Input
from .supervisor import Supervisor
from .pyramidal import Pyramidal
from .inter import Inter
from .output import Output
from .bias import Bias
from .connect import Connector
from . import PrinterMixin
from ..common import ObjectTypes

class LayerBase(PrinterMixin):
    def __init__(self, genn_model, size, level, label):
        self.size = size
        self.views = {}
        self.source = None
        self.label = label
        self.level = level
        self.pops = {}
        self.synapses = {}
        self.genn_model = genn_model
        self.pop = None
        self.var_views = {}

    def connect(self, source, params):
        pass

    def __build(self, params):
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "{}\n{}".format(
             self.label,
             self._str_params(2))
        return s

    def setup_global_param_views(self):
        for p in ObjectTypes:
            objs = self.synapses if p == ObjectTypes.SYNAPSES else self.pops
            vd = {}
            for k in objs:
                objs[k].setup_global_views()
                vd[k] = objs[k].views

            self.views[p] = vd

    def set_global_param_all(self, param, val):
        for objs in [self.pops, self.synapses]:
            for k in objs:
                if param in objs[k].views:
                    objs[k].set_global(param, val)

    def setup_variable_views(self):
        for p in ObjectTypes:
            objs = self.synapses if p == ObjectTypes.SYNAPSES else self.pops
            vd = {}
            for k in objs:
                objs[k].setup_variable_views()
                vd[k] = objs[k].var_views

            self.var_views[p] = vd

    def set_variable_value(self, target, var, val):
        objs = self._find_objs_for_var_target(target)
        objs[target].set_variable(var, val)

    def _print_global_param(self, param):
        print("Layer: {}".format(self.label))
        for k in self.pops:
            self.pops[k]._print_global_param(param)

        for k in self.synapses:
            self.synapses[k]._print_global_param(param)


class InputLayer(LayerBase):
    def __init__(self, genn_model, size, params):
        level = 0
        label = "InputLayer ({}) at {}".format(size, level)

        super().__init__(genn_model, size, level, label)

        self.__build(params)

    def __build(self, params):
        _size = self.size
        _label = 'input_layer_0'
        _model = self.genn_model
        _rates = params['rates']
        _params = params['params']
        _ini = params['ini']
        self.pops['input'] = Input(_model, _size, _rates, _label,
                                   _params, _ini)
        self.pop = self.pops['input']

    def set_rates_for_stage(self, stage):
        self.pop.set_rate_for_stage(stage, self.pop.rates)

    def connect(self, source, params):
        raise Exception("Input layers do not accept connections:"
                        "\n{} -> {}".format(source, self))


class SupervisorLayer(LayerBase):
    def __init__(self, genn_model, size, params, level):
        label = "SupervisorLayer ({}) {}".format(size, level)

        super().__init__(genn_model, size, level, label)
        self.__build(params)

    def __build(self, params):
        _size = self.size
        _model = self.genn_model
        _level = self.level
        _label = 'supervisor_layer_{}'.format(_level)
        _rates = params['rates']
        _params = params['params']
        _ini = params['ini']
        self.pops['supervisor'] = Supervisor(_model, _size, _rates, _label,
                                             _params, _ini, _level)
        self.pop = self.pops['supervisor']

    def connect(self, source, params):
        raise Exception("Supervisor layers do not accept connections:"
                        "\n{} -> {}".format(source, self))


class HiddenLayer(LayerBase):
    def __init__(self, genn_model, n_pyramidal, n_inter, params, level):
        self.n_inter = n_inter
        label = "HiddenLayer ({} pyr, {} inter) at {}".format(
                 n_pyramidal, n_inter, level)
        self.params = params
        self.inter = None
        super().__init__(genn_model, n_pyramidal, level, label)

        self.__build(params)

    def __build(self, params):
        _size = self.size
        _label = self.label
        _model = self.genn_model
        _level = self.level

        _pyr_params = params['pyramidal']
        print(_pyr_params)
        args = {
            'params': _pyr_params['params'],
            'init': _pyr_params['ini'],
            'label': 'hidden_layer_{}_pyramidal'.format(_level),
            'activation': params['activation'],
        }

        self.pops['pyramidal'] = Pyramidal(_model, _size, _level, **args)
        self.pop = self.pops['pyramidal']

        _size = self.n_inter
        _inn_params = params['inter']
        args = {
            'params': _inn_params['params'],
            'init': _inn_params['ini'],
            'label': 'hidden_layer_{}_inter'.format(_level),
            'activation': params['activation'],
        }

        self.pops['inter'] = Inter(_model, _size, _level, **args)
        self.inter = self.pops['inter']

    def lateral_connect(self, next_layer, params):
        _model = self.genn_model
        k = 'p2i'
        _ini = params[k]['ini']
        _params = params[k].get('params', {})
        _globals = params[k].get('global', {})
        p2i = Connector(_model, self.pop, self.inter, _params, _ini,
                        global_params=_globals)
        self.synapses[k] = p2i

        k = 'i2p'
        _ini = params[k]['ini']
        _params = params[k].get('params', {})
        _globals = params[k].get('global', {})
        i2p = Connector(_model, self.inter, self.pop, _params, _ini,
                        global_params=_globals)

        self.synapses[k] = i2p

    def bias_connect(self, bias_pop, params):
        if not (isinstance(bias_pop, Bias)):
            raise Exception("Unsupported source population for bias_connect")

        _model = self.genn_model
        _ini = params['to_pyramidal']['ini']
        _params = params['to_pyramidal'].get('params', {})
        _globals = params['to_pyramidal'].get('global', {})
        key = 'bias_to_pyr'
        c = Connector(_model, bias_pop, self.pop, _params, _ini,
                      global_params=_globals)
        self.synapses[key] = c

        _ini = params['to_inter']['ini']
        _params = params['to_inter'].get('params', {})
        _globals = params['to_inter'].get('global', {})
        key = 'bias_to_inter'
        c = Connector(_model, bias_pop, self.inter, _params, _ini,
                      global_params=_globals)
        self.synapses[key] = c

    def make_i2p_static(self):
        self.synapses['i2p'].views['learning_on'][:] = 0

    def connect(self, source, params):
        if not (isinstance(source, HiddenLayer) or
                isinstance(source, InputLayer) or
                isinstance(source, OutputLayer)):
            raise Exception("Connecting with Hidden layers is only allowed for"
                            "Hidden, Output or Input layer types:"
                            "\n{} -> {}".format(source, self))

        _model = self.genn_model
        _ini = params['ini']
        _params = params.get('params', {})
        _globals = params.get('global', {})
        _conn_init = None
        inter_c = None
        if isinstance(source, HiddenLayer):
            if source.level < self.level:
                key = 'ff'
            elif source.level > self.level:
                key = 'fb'
                # Higher pyramidal to lower inter
                inter_c = Connector(_model, source.pop, self.inter,
                                    {},
                                    {'g': 1.0},
                                    conn_init={'init': 'OneToOne', 'params': {}})
                self.synapses['fbi'] = inter_c

            else:
                raise Exception("Hidden layer types need to have different "
                                "levels to connect:\n{} -> {}".format(
                                 source, self))
        elif isinstance(source, OutputLayer):
            if source.level <= self.level:
                raise Exception("Output layer should be at least one level "
                                "above current layer:\n{} -> {}".format(
                                 source, self))
            key = 'fb'
            # Output to lower inter
            inter_c = Connector(_model, source.pop, self.inter,
                                {},
                                {'g': 1.0},
                                conn_init={'init': 'OneToOne', 'params': {}})
            self.synapses['fbi'] = inter_c
        else:
            if source.level >= self.level:
                raise Exception("Input layers should have level (0) bellow "
                                "the target Hidden:\n{} -> {}".format(
                                 source, self))
            key = 'ff'

        c = Connector(_model, source.pop, self.pop, _params, _ini,
                      global_params=_globals)
        self.synapses[key] = c


class OutputLayer(LayerBase):
    def __init__(self, genn_model, size, params, level):
        label = "OutputLayer ({}) at {}".format(size, level)
        super().__init__(genn_model, size, level, label)
        self.params = params
        self.__build(params)

    def __build(self, params):
        _size = self.size
        _label = self.label
        _model = self.genn_model
        _level = self.level

        args = {
            'params': params['params'],
            'ini': params['ini'],
            'label': 'out_layer_{}'.format(_level),
            'global_params': params['global'],
            'activation': params['activation'],
        }

        self.pops['output'] = Output(_model, _size, _level, **args)
        self.pop = self.pops['output']

    def connect(self, source, params):
        if not (isinstance(source, HiddenLayer) or
                isinstance(source, SupervisorLayer)):
            raise Exception("Connecting with Output layer is only allowed for"
                            "Hidden or Supervisor layer types:"
                            "\n{} -> {}".format(source, self))

        _model = self.genn_model
        _ini = params['ini']
        _params = params.get('params', {})
        _globals = params.get('global', {})
        pyr = isinstance(source, HiddenLayer)
        key = 'ff' if pyr else 'fb'
        _conn_ini = {'init': 'OneToOne', 'params': {}} if not pyr else None
        c = Connector(_model, source.pop, self.pop, _params, _ini,
                       global_params=_globals,
                       conn_init=_conn_ini)
        self.synapses[key] = c

    def bias_connect(self, bias_pop, params):
        if not(isinstance(bias_pop, Bias)):
            raise Exception("Unsupported source population for bias_connect")

        _model = self.genn_model
        _ini = params['ini']
        _params = params.get('params', {})
        _globals = params.get('global', {})
        key = 'bias'
        c = Connector(_model, bias_pop, self.pop, _params, _ini,
                      global_params=_globals)
        self.synapses[key] = c



