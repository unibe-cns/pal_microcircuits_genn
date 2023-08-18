import numpy as np
from .layers import (InputLayer, HiddenLayer, OutputLayer, SupervisorLayer)
from .bias import Bias
from . import PrinterMixin
from pygenn.genn_model import GeNNModel
from .recorder import Recorder
import sys
import time
from numpy.testing import assert_array_almost_equal
from microcircuits.genn_models.common import (activation_templates,
                                              act_numpy, debug_prints,
                                              Stages)
from microcircuits.genn_models.defaults import (default_activation,
                                                default_activation_name,
                                                default_act_numpy)
from pprint import pprint
from warnings import warn
from scipy import stats


class Network(PrinterMixin):
    def __init__(self, description, **kwargs):
        self.desc = description
        self.dims = description.get('dims', None)
        if self.dims is None:
            raise Exception("Network requires a size per layer in a list")

        bias_desc = description.get('bias', {})
        self.bias_on = bias_desc.get('on', False)
        self.bias_value = bias_desc.get('val', 0.0)
        self.start_t = None

        act_name = description.get('activation_template',
                                   default_activation_name)
        self.activation_tmpl = activation_templates[act_name]

        if 'activation_template' in description:
            if 'activation_numpy' in description:
                act_npy = description['activation_numpy']
            elif act_name in act_numpy:
                act_npy = act_numpy[act_name]
            else:
                raise Exception("Could not find the numpy version of the given "
                                "activation in the microcircuit package's "
                                "list of activations nor in the Network "
                                "creation arguments. Please provide both an "
                                "activation template and a numpy version.")

            self.activation_numpy = act_npy
        else:
            self.activation_numpy = default_act_numpy

        backend = description.get("backend", "SingleThreadedCPU")
        print("Running simulation on {} backend".format(backend), flush=True)
        self.name = description.get("name", "genn_microcircuit_network")
        self.genn_model = GeNNModel("float", self.name, backend=backend)
        self.genn_model.dT = description.get('dt', 1.0)
        self.reflect = description.get('start_in_self_predict', True)
        self.print_steps = description.get('print_every_n_recs', 1)
        self.views = {}
        self.var_views = {}

        ls = [
            InputLayer(self.genn_model, self.dims[0],
                       self._parse_input_params())
        ]
        for level in range(1, len(self.dims) - 1):
            ls.append(
                HiddenLayer(self.genn_model, self.dims[level],
                            self.dims[level + 1],
                            self._parse_hidden_params(),
                            level
                            )
            )

        ls.append(
            OutputLayer(self.genn_model, self.dims[-1],
                        self._parse_output_params(),
                        len(self.dims) - 1)
        )
        self.layers = ls

        self.bias_neuron = (Bias(self.genn_model, self.bias_value)
                            if self.bias_on else None)

        self.weights = {}

        print("Feed-forward connections")
        for idx, pre in enumerate(self.layers[:-1]):
            print(idx, idx + 1)
            post = self.layers[idx + 1]

            p = self._parse_conn_params(pre, post)
            post.connect(pre, p)

        print("Feed-back connections")
        for idx in range(len(self.layers) - 1, 1, -1):
            print(idx, idx - 1)
            pre = self.layers[idx]
            post = self.layers[idx - 1]
            p = self._parse_conn_params(pre, post)
            post.connect(pre, p)

        print("Lateral connections")
        for idx, pre in enumerate(self.layers[:-1]):
            if idx == 0:
                continue
            print(idx)
            post = self.layers[idx + 1]
            p = self._parse_lateral_conn_params(pre, post)
            pre.lateral_connect(post, p)

        self.check_self_predict_start()

        if self.bias_on:
            print("Bias connections")
            pre = self.bias_neuron
            for idx, post in enumerate(self.layers[:-1]):
                if idx == 0:
                    continue
                print(idx)

                p = self._parse_bias_conn_params(pre, post)
                post.bias_connect(pre, p)

        ls.append(
            SupervisorLayer(self.genn_model, self.dims[-1],
                            self._parse_supervisor_params(),
                            len(self.dims))
        )

        post, pre = self.layers[-2:]
        post.connect(pre, self._parse_conn_params(pre, post))

        # print(self)
        # print("Printing layers from Network")
        # for l in self.layers:
        #     print("===============================================")
        #     print(l.label)
        #     for k, p in l.pops.items():
        #         print(f"{k}\n{p}")
        #     for k, s in l.synapses.items():
        #         print(f"{k}\n{s}")
            # if len(l.synapses) == 0:
            #     continue
            # print("------------------------------------------")
            # print("---------------   synapses   -------------")
            # for k in l.synapses:
            #     print("------------------------------------------")
            #     print(k)
            #     print(l.synapses[k])

        self.recorder = Recorder(self.genn_model, self.layers)
        self.enable_recording = True

    def check_self_predict_start(self):
        reflect = self.reflect
        if not reflect:
            return False

        ws = self.weights
        for lyr_i in range(1, len(self.layers) - 2):
            w_ff = ws[lyr_i][lyr_i + 1][0]
            w_fb = ws[lyr_i + 1][lyr_i][0]
            w_p2i = ws[lyr_i][lyr_i][0]
            w_i2p = ws[lyr_i][lyr_i][1]

            # print(w_ff.shape, w_p2i.shape)
            # print(w_fb.shape, w_i2p.shape)
            print("Checking reflected weights for layer {}".format(lyr_i))
            assert_array_almost_equal(w_ff, w_p2i)
            assert_array_almost_equal(w_fb, -w_i2p)

    def _parse_input_params(self):
        d = self.desc
        params = {
            'period': d['t_pattern'],
            'tau': d['tau_0'],
        }
        ini = {}
        rates = d['input_rates']
        return {'params': params, 'ini': ini, 'rates': rates}

    def _parse_hidden_params(self):
        d = self.desc
        ppar = {
            'g_leak': d['gl'],
            'g_basal': d['gb'],
            'g_apical': d['ga'],
            'le_active': d['le_active'],
            'sigma_noise': d['noise_sig'],
            'tau_noise': d['noise_tau'],
            'tau_highpass': d['highpass_tau'],
        }
        pini = {}

        ipar = {
            'g_leak': d['gl'],
            'g_soma': d['gsom'],
            'g_dendrite': d['gd'],
            'le_active': d['le_active'],
        }
        iini = {}

        p = {
            'pyramidal': {'params': ppar, 'ini': pini},
            'inter': {'params': ipar, 'ini': iini},
            'activation': self.activation_tmpl
        }
        return p

    def _parse_output_params(self):
        d = self.desc
        par = {
            'g_leak': d['gl'],
            'g_basal': d['gb'],
            'g_target': d['gtar'],
            'tau_highpass': d['highpass_tau'],
            'le_active': d['le_active'],
        }
        ini = {}
        g = {'learning_on': 1}
        return {'params': par, 'ini': ini,
                'global': g, 'activation': self.activation_tmpl}

    def _parse_supervisor_params(self):
        d = self.desc
        par = {
            'mute_t': d['out_lag'],
            'period': d['t_pattern'],
            'tau': d['tau_0'],
            'base_rate': d['sim_config'].get('ulow', 0)
        }
        ini = {}
        r = d['output_rates']

        if isinstance(r, str) and r == 'ideal':
            r = self.calculate_output_rates()

        return {'params': par, 'ini': ini, 'rates': r}

    def calculate_output_rates(self):
        in_pop = self.layers[0].pop
        out = {}
        for stage in in_pop.rates:
            rate = in_pop.rates[stage]
            in_s = self.layers[0].size
            x = rate.reshape((in_s, -1))
            out_idx = len(self.layers) - 1

            for lidx in range(1, len(self.layers)):
                in_s = self.layers[lidx - 1].size
                out_s = self.layers[lidx].size

                lyr = self.layers[lidx]
                if lidx < out_idx:
                    gl = lyr.params['pyramidal']['params']['g_leak']
                    gb = lyr.params['pyramidal']['params']['g_basal']
                    ga = lyr.params['pyramidal']['params']['g_apical']
                else:
                    gl = lyr.params['params']['g_leak']
                    gb = lyr.params['params']['g_basal']
                    ga = 0

                s = gb / (gl + gb + ga)

                w = lyr.synapses['ff'].init['g'].copy().reshape((in_s, out_s)).T
                x = s * np.matmul(w, x)
                if lidx < out_idx:
                    x = self.activation_numpy(x)

                del w

            out[stage] = x

        return out

    def _store_weights(self, g, pre, post):
        wpre = self.weights.get(pre.level, {})
        wl = wpre.get(post.level, [])
        wl.append(g)
        wpre[post.level] = wl
        self.weights[pre.level] = wpre

    def _parse_conn_params(self, pre, post):
        if isinstance(pre, HiddenLayer):
            if isinstance(post, HiddenLayer):
                return self._parse_hidden_to_hidden(pre, post)
            elif isinstance(post, OutputLayer):
                return self._parse_hidden_to_out(pre, post)
        elif isinstance(pre, OutputLayer):
            if isinstance(post, HiddenLayer):
                return self._parse_out_to_hidden(pre, post)
        elif isinstance(pre, InputLayer):
            if isinstance(post, HiddenLayer):
                return self._parse_in_to_hidden(pre, post)
        elif isinstance(pre, SupervisorLayer):
            if isinstance(post, OutputLayer):
                return self._parse_sup_to_out(pre, post)
        raise Exception("Not supported connectivity:\n"
                        "{} to {}".format(pre, post))

    def _conn_is_plastic(self, pre, post):
        apt_source = (isinstance(pre, InputLayer) or
                      isinstance(pre, HiddenLayer))
        post_higher = pre.level < post.level
        return apt_source and post_higher

    def _get_init_conn_params(self, pre, post, glim):
        d = self.desc
        ff = pre.level < post.level
        nrows = pre.pop.size
        ncols = post.pop.size
        if type(glim) == list:
            g = np.random.uniform(glim[0], glim[1], size=(nrows, ncols))
        else:
            g = np.random.uniform(-glim, glim, size=(nrows, ncols))
        ini = {'g': g}
        par = {}
        if ff:
            ini['delta'] = 0
            par['eta'] = d['eta']['up'][pre.level]
            par['tau'] = d['tau_w_ff']
        else:
            par['eta'] = d['eta']['down'][post.level]
            par['alpha'] = d['alpha']

        # before only forward weights and lateral weights were plastic
        #is_plastic = self._conn_is_plastic(pre, post)
        # now also backward weights are plastic
        is_plastic = True
        glb = {'learning_on': 1} if is_plastic else None
        self._store_weights(g, pre, post)

        return {'params': par, 'ini': ini, 'global': glb}

    def _parse_sup_to_out(self, pre, post):
        ini = {'g': 1}
        par = {}
        glb = {'learning_on': 1}
        return {'params': par, 'ini': ini, 'global': glb}

    def _parse_out_to_hidden(self, pre, post):
        d = self.desc
        glim = d['init_weights']['down']
        return self._get_init_conn_params(pre, post, glim)

    def _parse_hidden_to_hidden(self, pre, post):
        d = self.desc
        ff = pre.level < post.level
        key = 'up' if ff else 'down'
        glim = d['init_weights'][key]
        return self._get_init_conn_params(pre, post, glim)

    def _parse_hidden_to_out(self, pre, post):
        d = self.desc
        glim = d['init_weights']['up']

        return self._get_init_conn_params(pre, post, glim)

    def _parse_in_to_hidden(self, pre, post):
        d = self.desc
        glim = d['init_weights']['up']
        return self._get_init_conn_params(pre, post, glim)

    def _parse_bias_conn_params(self, pre, post):
        def _params(key, pre_size, post_size, layer):
            d = self.desc
            glim = d['init_weights'][key]
            nrows = pre_size
            ncols = post_size
            if type(glim) == list:
                g = np.random.uniform(glim[0], glim[1], size=(nrows, ncols))
            else:
                g = np.random.uniform(-glim, glim, size=(nrows, ncols))
            ini = {
                'g': g,
                'delta': 0
            }
            par = {
                'eta': d['eta'][key][layer - 1],
                'tau': d['tau_w_ff']
            }

            glb = {'learning_on': 1}
            # self._store_weights(g, pre, post)

            return {'params': par, 'ini': ini, 'global': glb}

        if isinstance(post, HiddenLayer):
            # bias to interneuron
            key = 'ip'
            b2i = _params(key, pre.size, post.inter.size, post.level)

            # bias to pyr
            key = 'up'
            b2p = _params(key, pre.size, post.pop.size, post.level)

            return {'to_inter': b2i, 'to_pyramidal': b2p}

        elif isinstance(post, OutputLayer):
            key = 'up'
            b2o = _params(key, pre.size, post.pop.size, post.level)

            return b2o

    def _parse_lateral_conn_params(self, layer, next):
        d = self.desc
        self_pred = self.reflect

        # pyramidal to interneuron
        key = 'ip'
        glim = d['init_weights'][key]
        nrows, ncols = layer.pop.size, layer.inter.size
        if self_pred:
            g = self.weights[layer.level][next.level][0].copy()
        else:
            if type(glim) == list:
                g = np.random.uniform(glim[0], glim[1], size=(nrows, ncols))
            else:
                g = np.random.uniform(-glim, glim, size=(nrows, ncols))
        ini = {'g': g, 'delta': 0}
        par = {
            'eta': d['eta'][key][layer.level - 1],
            'tau': d['tau_w_ip']
        }
        glb = {'learning_on': 1}
        self._store_weights(g, layer, layer)
        p2i = {'ini': ini, 'params': par, 'global': glb}

        # interneuron to pyramidal
        key = 'pi'
        glim = d['init_weights'][key]
        nrows, ncols = layer.inter.size, layer.pop.size
        if self_pred:
            g = -(self.weights[next.level][layer.level][0].copy())
        else:
            if type(glim) == list:
                g = np.random.uniform(glim[0], glim[1], size=(nrows, ncols))
            else:
                g = np.random.uniform(-glim, glim, size=(nrows, ncols))
        ini = {'g': g, 'delta': 0}
        par = {
            'eta': d['eta'][key][layer.level - 1],
            'tau': d['tau_w_pi']
        }
        glb = {'learning_on': 1}
        self._store_weights(g, layer, layer)
        i2p = {'ini': ini, 'params': par, 'global': glb}

        return {'p2i': p2i, 'i2p': i2p}

    def __str__(self):
        s = """
        \n**************************************
        Network ({}):
        {}
        \n**************************************\n
        """.format(self.dims,
                   self._str_params())
        return s

    def set_learning(self, set_on):
        for lyr_idx in range(1, len(self.layers) - 1):
            lyr = self.layers[lyr_idx]
            lyr.set_global_param_all('learning_on', set_on)
        if debug_prints:
            self._print_global_param('learning_on')

    def turn_learning_on(self):
        if debug_prints:
            print("\n\t************** Turn learning ON **************", flush=True)
        self.set_learning(1)

    def turn_learning_off(self):
        if debug_prints:
            print("\n\t************** Turn learning OFF *************", flush=True)
        self.set_learning(0)

    def setup_global_param_views(self):
        gvs = {}
        for lyr_idx, lyr in enumerate(self.layers):
            lyr.setup_global_param_views()
            gvs[lyr_idx] = lyr.views

        self.views = gvs

    def setup_variable_views(self):
        vvs = {}
        for lyr_idx, lyr in enumerate(self.layers):
            lyr.setup_variable_views()
            vvs[lyr_idx] = lyr.var_views

        self.var_views = vvs

    def disconnect_teaching(self):
        if debug_prints:
            print("\t........ Disabling Teaching Signal ........")
        lyr = len(self.layers) - 2
        self.views[lyr]['neurons']['output']['learning_on'][:] = 0
        self.views[lyr]['synapses']['fb']['learning_on'][:] = 0
        if debug_prints:
            pprint(self.views)

    def reconnect_teaching(self):
        if debug_prints:
            print("\t........ Enabling Teaching Signal ........")
        lyr = len(self.layers) - 2
        self.views[lyr]['neurons']['output']['learning_on'][:] = 1
        self.views[lyr]['synapses']['fb']['learning_on'][:] = 1
        if debug_prints:
            pprint(self.views)

    def make_inter_to_pyramidal_static(self):
        if debug_prints:
            print("\t+++++++++++++++++++++++++++++++++"
                  " Making i2p connections non-plastic "
                  "+++++++++++++++++++++++++++++++++")

        for lyr_idx in range(1, len(self.layers) - 2):
            self.layers[lyr_idx].make_i2p_static()

        if debug_prints:
            pprint(self.views)

    def _is_start_of_validation_loops(self, loop_idx, is_learning_on,
                                      last_train_switch, train_idx, valid_idx, epoch_idx):
        didx = (loop_idx + valid_idx)
        return ((didx % epoch_idx) == 0)

    def _is_start_of_training_loops(self, loop_idx, is_learning_on,
                                    last_train_switch, train_idx, valid_idx, epoch_idx):
        return ((loop_idx % epoch_idx) == 0)

    def _delay_learning(self, delay_learn_steps, loop_idx, sample_idx,
                        is_learning_on, disable_teaching, disable_learning):

        if delay_learn_steps > 0:
            txt = "\n\tDelaying learning for {} steps".format(delay_learn_steps)
            didx = loop_idx - delay_learn_steps
            if is_learning_on and (loop_idx % sample_idx) == 0:
                if debug_prints:
                    print(txt)
                    print("\tSETTING OFF")
                self.turn_learning_off()
                self.disconnect_teaching()

            elif is_learning_on and (didx % sample_idx) == 0:
                if debug_prints:
                    print(txt)
                    print("\tSETTING ON")

                if not disable_learning:
                    self.turn_learning_on()

                if disable_teaching:
                    self.disconnect_teaching()
                else:
                    self.reconnect_teaching()

                if self.reflect:
                    # if the backward weights are static and we start in selfpred state
                    if not any(eta > 0 for eta in self.desc['eta']['down']):
                        print('Setting eta_i2p to zero, delay learning')
                        self.make_inter_to_pyramidal_static()

    def _evaluate_validation(self, test=False):
        if not self.enable_recording:
            return 1, 0

        conf = self.desc['sim_config']
        idx_rec = conf['idx_rec']
        if test:
            idx_val = conf['idx_test']
        else:
            idx_val = conf['idx_valid']
        idx_smp = conf['idx_sample']

        n_per_sample = idx_smp // idx_rec
        n_val = idx_val // idx_rec
        rec = self.recorder

        if test:
            idx_rate = rec.rates_idx[Stages.TESTING]
            cache = rec.cache[Stages.TESTING]
        else:
            idx_rate = rec.rates_idx[Stages.VALIDATION]
            cache = rec.cache[Stages.VALIDATION]
        cache_keys = sorted(cache.keys())
        out_i = cache_keys[-2]
        sup_i = cache_keys[-1]

        # analysis is done after recording, so we need to look back?
        out_v = cache[out_i]['output']['v_brev'][idx_rate - n_val: idx_rate, :]
        #out_v = cache[out_i]['output']['V'][idx_rate - n_val: idx_rate, :]
        sup_v = cache[sup_i]['supervisor']['rate'][idx_rate - n_val: idx_rate, :]
        mse = np.mean(np.sqrt((out_v - sup_v) ** 2))

        out_m = np.vstack([np.mean(out_v[se + 1:se + n_per_sample, :], axis=0)
                           for se in range(0, n_val, n_per_sample)])
        sup_m = np.vstack([np.max(sup_v[se + 1:se + n_per_sample, :], axis=0)
                           for se in range(0, n_val, n_per_sample)])

        pred = np.argmax(out_m, axis=1)
        real = np.argmax(sup_m, axis=1)
        acc = np.mean(pred == real)

        print("\nMean Square Error = {:10.3f}".format(mse))
        print("Accuracy = {:6.2f}%".format(100 * acc))
        return mse, acc

    def _print_global_param(self, p):
        pprint(self.views)
        # for lyr in self.layers:
        #     lyr._print_global_param(p)

    def reset_temporal_counts(self):
        self.genn_model.timestep = 0
        self.genn_model.t = 0.0

    def reset_rate_players(self, stage, shuffled_indices=None):
        self.layers[0].pop.reset_player(stage, shuffled_indices=shuffled_indices)
        self.layers[-1].pop.reset_player(stage, shuffled_indices=shuffled_indices)

    def _run(self, stage, run_steps, index, total_steps, learning_on,
             weight_update_index, rate_update_index, write_index,
             never_use_teach_signal, disable_learning):
        desc = self.desc
        dt = desc['dt']
        sample_index = desc['sim_config']['idx_sample']
        steps_per_epoch = desc['sim_config']['idx_epoch']
        lag_index = int(self.desc['learning_lag'] / dt)
        local_index = 0
        print_index = 0
        start_t = self.start_t
        _model = self.genn_model
        self.reset_temporal_counts()
        if desc['sim_config']['shuffle_samples']:
            n_indices = desc['input_rates'][stage].shape[1]
            indices = np.arange(n_indices, dtype=np.int)
            np.random.shuffle(indices)
            self.reset_rate_players(stage, shuffled_indices=indices)
        else:
            self.reset_rate_players(stage)
        epoch_idx = index // steps_per_epoch
        # used the FLOAT runtime here before, bad idea ...
        while local_index < run_steps:
        # while _model.t < run_time:
            print_index += 1
            pc = 100.0 * (float(index) / float(total_steps))

            self._delay_learning(lag_index, local_index, sample_index, learning_on,
                                 never_use_teach_signal, disable_learning)
            _model.step_time()

            update_rates = (index % rate_update_index == 0)
            update_weights = ((index % weight_update_index == 0) and learning_on)
            write_out = index > 0 and (index % write_index == 0)
            if update_rates:
                if print_index >= self.print_steps:
                    print_index = 0
                    elapsed_time = time.process_time() - start_t
                    sys.stdout.write(
                        "\rSimulating {:7.3f}%\tLocal {:10d}\tGlobal {:10d}\tEpoch {:6d}\tElapsed {:10d}s\n".format(
                            pc, local_index, index, epoch_idx, int(elapsed_time)))
                    sys.stdout.flush()
                    # print("\nglobal {}\tlocal {}".format(index, local_index))

            if self.enable_recording:
                self.recorder.update(stage, update_rates, update_weights, write_out)


            local_index += 1
            index += 1

        return index

    def run_train(self, run_steps, index, total_steps, weight_update_index,
                  rate_update_index, write_index, never_use_teach_signal,
                  disable_learning):
        if disable_learning:
            self.turn_learning_off()
        else:
            self.turn_learning_on()

        if never_use_teach_signal:
            self.disconnect_teaching()
        else:
            self.reconnect_teaching()

        if self.reflect:
            # if the backward weights are static and we start in selfpred state
            if not any(eta > 0 for eta in self.desc['eta']['down']):
                print('Setting eta_i2p to zero, run_train')
                self.make_inter_to_pyramidal_static()

        return self._run(Stages.TRAINING, run_steps, index, total_steps, True,
                         weight_update_index, rate_update_index, write_index,
                         never_use_teach_signal, disable_learning)


    def run_validation(self, run_steps, index, total_steps, weight_update_index,
                       rate_update_index, write_index):

        self.turn_learning_off()
        self.disconnect_teaching()

        return self._run(Stages.VALIDATION, run_steps, index, total_steps, False,
                         weight_update_index, rate_update_index, write_index,
                         True, True)


    def run_test(self, run_steps, index, total_steps, weight_update_index,
                 rate_update_index, write_index):

        self.turn_learning_off()
        self.disconnect_teaching()

        return self._run(Stages.TESTING, run_steps, index, total_steps, False,
                         weight_update_index, rate_update_index, write_index,
                         True, True)


    def run(self, **kwargs):
        recording_config = kwargs.get('recording_config', None)

        if recording_config is None:
            recording_config = self.desc.get('record', None)

        if recording_config is None:
            warn("Running without recording, you will not be able "
                 "to analyze results.")
            self.enable_recording = False

        never_use_teach_signal = kwargs.get('never_use_teach_signal', False)
        disable_learning = kwargs.get('disable_learning', False)

        _config = self.desc['sim_config']
        _model = self.genn_model

        _model.build()
        _model.load()

        self.setup_global_param_views()
        self.setup_variable_views()

        if self.reflect:
            # if the backward weights are static and we start in selfpred state
            if not any(eta > 0 for eta in self.desc['eta']['down']):
                print('Setting eta_i2p to zero, run')
                self.make_inter_to_pyramidal_static()

        wu_idx = recording_config['update_indices']['weights']
        ru_idx = recording_config['update_indices']['rates']
        write_idx = recording_config['update_indices']['write']

        dt = self.desc['dt']
        n_epoch = _config['n_epochs']
        pat_t = _config['pat_t']
        n_train = _config['n_train'] * _config['n_class']
        n_valid = _config['n_valid'] * _config['n_class']
        n_test = _config['n_test'] * _config['n_class']
        total_sim_time = _config['sim_time'] + (n_valid * pat_t) + (n_test * pat_t)
        total_sim_steps = int(total_sim_time / dt)
        train_run_time = n_train * pat_t
        train_run_steps = _config['idx_train']
        valid_run_time = n_valid * pat_t
        valid_run_steps = _config['idx_valid']
        test_run_time = n_test * pat_t
        test_run_steps = _config['idx_test']

        if self.enable_recording:
            self.recorder.init(recording_config)

        if never_use_teach_signal:
            self.disconnect_teaching()
        if disable_learning:
            self.turn_learning_off()

        self.start_t = time.process_time()
        loop_idx = np.int(0)
        all_acc = []
        all_mse = []
        for epoch_count in range(n_epoch):
            print('Epoch', epoch_count)
            # if set_wBackw_to_wTransp selected: set backward weights to backprop-mode
            # i.e. transposed of forward weights
            # additionally the inter-to-pyr weights need to be adjusted accordingly
            # TODO: properly handle more than one hidden layer
            if self.desc['set_wBackw_to_wTransp']:
                print('WARNING: setting backward weigths to transposed of forward')
                n_hid = self.desc['dims'][-2]
                n_top = self.desc['dims'][-1]
                w_forw = self.layers[2].synapses['ff'].var_views['g'].copy().reshape(n_hid, n_top)
                w_transp = np.transpose(w_forw)
                self.layers[1].synapses['fb'].set_variable('g', w_transp.flatten())
                self.layers[1].synapses['i2p'].set_variable('g', -1 * w_transp.flatten())
                w_fb = self.layers[1].synapses['fb'].var_views['g'].copy()
                w_i2p = self.layers[1].synapses['i2p'].var_views['g'].copy()
                if np.any(np.abs(w_fb + w_i2p) > 0):
                    print(np.abs(w_fb + w_i2p))
                    print(w_fb)
                    print(w_i2p)
                    raise RuntimeError

            if self.desc['debug_mode']:
                w_ff_1_pre = self.layers[1].synapses['ff'].var_views['g'].copy()
                w_ff_2_pre = self.layers[2].synapses['ff'].var_views['g'].copy()
                w_p2i_1_pre = self.layers[1].synapses['p2i'].var_views['g'].copy()
                w_i2p_1_pre = self.layers[1].synapses['i2p'].var_views['g'].copy()
                w_fb_pre = self.layers[1].synapses['fb'].var_views['g'].copy()

            print("\n>>>>>>>>>>>>> start of VALIDATION <<<<<<<<<<<<<<", flush=True)
            loop_idx = self.run_validation(
                            valid_run_steps, loop_idx, total_sim_steps,
                            wu_idx, ru_idx, write_idx)
            mse, acc = self._evaluate_validation()
            all_mse.append(mse)
            all_acc.append(acc)

            if self.desc['debug_mode']:
                w_ff_1_post = self.layers[1].synapses['ff'].var_views['g'].copy()
                w_ff_2_post = self.layers[2].synapses['ff'].var_views['g'].copy()
                w_p2i_1_post = self.layers[1].synapses['p2i'].var_views['g'].copy()
                w_i2p_1_post = self.layers[1].synapses['i2p'].var_views['g'].copy()
                w_fb_post = self.layers[1].synapses['fb'].var_views['g'].copy()
                if np.any(np.abs(w_ff_1_post - w_ff_1_pre) > 0) or \
                        np.any(np.abs(w_ff_2_post - w_ff_2_pre) > 0) or \
                        np.any(np.abs(w_p2i_1_post - w_p2i_1_pre) > 0) or \
                        np.any(np.abs(w_fb_post - w_fb_pre) > 0) or \
                        np.any(np.abs(w_i2p_1_post - w_i2p_1_pre) > 0):
                    print('weight_diffs after valid epoch:')
                    print('w_ff_1', w_ff_1_post - w_ff_1_pre)
                    print('w_ff_2', w_ff_2_post - w_ff_2_pre)
                    print('w_p2i_1', w_p2i_1_post - w_p2i_1_pre)
                    print('w_i2p_1', w_i2p_1_post - w_i2p_1_pre)
                    print('w_fb', w_fb_post - w_fb_pre)
                    raise RuntimeError

            print(all_mse)
            print(all_acc)

            print("\n>>>>>>>>>>>>> start of TRAINING <<<<<<<<<<<<<<", flush=True)
            loop_idx = self.run_train(
                            train_run_steps, loop_idx, total_sim_steps,
                            wu_idx, ru_idx, write_idx, never_use_teach_signal,
                            disable_learning)

            # if reinforce selfpredict selected: reset weights to perfect selfpred after every training epoch
            # TODO: ensure correct setting in case of different neuron params
            # TODO: properly handle more than one hidden layer
            if self.desc['reinforce_self_predict']:
                print('WARNING: reinforcing selfpred state')
                self.layers[1].synapses['p2i'].set_variable('g', self.layers[2].synapses['ff'].var_views['g'].copy())
                w_ff_2 = self.layers[2].synapses['ff'].var_views['g'].copy()
                w_p2i = self.layers[1].synapses['p2i'].var_views['g'].copy()
                print(np.abs(w_ff_2 - w_p2i))
                if np.any(np.abs(w_ff_2 - w_p2i) > 0):
                    print(w_ff_2)
                    print(w_p2i)
                    raise RuntimeError

        print("\n>>>>>>>>>>>>> final VALIDATION <<<<<<<<<<<<<<")
        self.run_validation(valid_run_steps, loop_idx, total_sim_steps,
                            wu_idx, ru_idx, write_idx)

        mse, acc = self._evaluate_validation()
        all_mse.append(mse)
        all_acc.append(acc)
        print(all_mse)
        print(all_acc)

        print("\n>>>>>>>>>>>>> final TESTING <<<<<<<<<<<<<<")
        self.run_test(test_run_steps, loop_idx, total_sim_steps,
                      wu_idx, ru_idx, write_idx)
        mse, acc = self._evaluate_validation(test=True)
        if self.enable_recording:
            self.recorder.update(Stages.TESTING, True, True, True)

        return (time.process_time() - self.start_t)
