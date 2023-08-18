import os
from warnings import warn

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
#plt.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
import h5py
from os import path
import os
from pprint import pformat, pprint
from scipy import stats
import sys
#from numba import jit
from matplotlib import colors as _mcolors
from microcircuits.genn_models.common import Stages

# BASIC_COLORS = ('red', 'green', 'blue', 'cyan', 'magenta', 'orange')
import random
random.seed(19)
_banned_colors = ['snow', 'honeydew', 'azure', 'aliceblue', 'lavenderblush',
                  'seashell', 'linen', 'mistyrose', 'ivory']
BASIC_COLORS = [v for k, v in _mcolors.CSS4_COLORS.items()
                if ('white' not in v or
                    'cream' not in v or
                    'yellow' not in v or
                    'lavender' not in v or
                    v not in _banned_colors)]
random.shuffle(BASIC_COLORS)
BASIC_COLORS = tuple(BASIC_COLORS)
BASIC_COLORS = tuple([v for k, v in _mcolors.TABLEAU_COLORS.items()])

class Analyzer:
    MAX_PLOT_POINTS = 100000

    def __init__(self, meta_config, plots_alpha=0.3, linewidth=1):
        self.alpha = plots_alpha
        self.linewidth = linewidth
        self.config = meta_config
        self.h5_filename = self.config['record']['filename']
        self.base_dir = os.path.dirname(os.path.abspath(self.h5_filename)) + '/'
        self.h5f = h5py.File(self.h5_filename, 'r')
        self.paths = {}
        for stg in self.h5f:
            lyrs = {}
            for lyr in self.h5f[path.join(stg)]:
                pops = {}
                for pop in self.h5f[path.join(stg, lyr)]:
                    sigs = {signal: path.join(stg, lyr, pop, signal)
                            for signal in self.h5f[path.join(stg, lyr, pop)]}
                    pops[pop] = sigs
                lyrs[lyr] = pops
            self.paths[stg] = lyrs

        self._stages = sorted(self.paths.keys())
        self._sorted_layer_ids = sorted(self.paths[self._stages[0]].keys())
        pprint(self.paths)

        self._training_indices, self._train_epoch_indices = \
                                            self._get_indices_for_training()
        self._validation_indices, self._valid_epoch_indices = \
                                            self._get_indices_for_validation()

        self.error_val = None
        self.accuracy_val = None
        self._differences_input = {}

    @property
    def lw(self):
        return self.linewidth

    def _get_indices_for_validation(self):
        rec_idx = self.config['sim_config']['idx_rec']
        val_idx = self.config['sim_config']['idx_valid'] // rec_idx
        # add extra validation at the end
        n_epochs = self.config['sim_config']['n_epochs'] + 1
        total_idx = (val_idx * n_epochs)
        ids = [np.arange(start, start + val_idx, dtype='int')
               for start in range(0, total_idx, val_idx)]
        eids = np.asarray([i * len(x) for i, x in enumerate(ids)])
        return np.hstack(ids), eids

    def _get_indices_for_training(self):
        rec_idx = self.config['sim_config']['idx_rec']
        trn_idx = self.config['sim_config']['idx_train'] // rec_idx
        total_idx = (trn_idx * self.config['sim_config']['n_epochs'])
        ids = [np.arange(start, start + trn_idx)
               for start in range(0, total_idx, trn_idx)]
        eids = np.asarray([i * len(x) for i, x in enumerate(ids)])
        return np.hstack(ids), eids

    def get_signal_for_validation(self, signal_path):
        s = self.h5f[signal_path]
        return s

    def get_signal_for_testing(self, signal_path):
        s = self.h5f[signal_path]
        return s

    def get_signal_for_training(self, signal_path):
        s = self.h5f[signal_path]
        return s

    def get_full_signal(self, signal_path):
        return self.h5f[signal_path]

    @property
    def output_layer_id(self):
        # TODO: this fails if supervisor signal is not recorded
        return self._sorted_layer_ids[-2]

    @property
    def supervisor_layer_id(self):
        return self._sorted_layer_ids[-1]

    def get_out_and_sup_signals(self, which_stage='validation'):
        if which_stage == 'validation':
            f = self.get_signal_for_validation
            stage = Stages.VALIDATION
        elif which_stage == 'testing':
            f = self.get_signal_for_testing
            stage = Stages.TESTING
        else:
            f = self.get_signal_for_training
            stage = Stages.TRAINING

        sup_id = self.supervisor_layer_id
        out_id = self.output_layer_id

        try:
            sup_path = [p for k, p in self.paths[stage][sup_id]['supervisor'].items()
                        if 'rate' in k][0]
        except:
            raise Exception('Can not find the "{}" signal for the "{}" '
                            'population, please set it as recordable in your '
                            'simulation'.format('rate', 'supervisor'))

        try:
            out_path = [p for k, p in self.paths[stage][out_id]['output'].items()
                        if 'v_brev' in k][0]
        except:
            raise Exception('Can not find the "{}" signal for the "{} '
                            'population", please set it as recordable in your '
                            'simulation'.format('V', 'output'))

        sup = f(sup_path)
        out = f(out_path)
        return out, sup

    def get_error_signal(self):
        rec_idx = self.config['sim_config']['idx_rec']
        n_per_epoch = self.config['sim_config']['idx_valid'] // rec_idx
        out, sup = self.get_out_and_sup_signals()

        diff = [out[si:si+n_per_epoch] - sup[si:si+n_per_epoch]
                for si in range(0, out.shape[0], n_per_epoch)]
        diff = [np.sqrt(np.mean((d)**2)) for d in diff]
        return diff

    def get_accuracy_signal(self):
        rec_idx = self.config['sim_config']['idx_rec']
        n_per_sample = self.config['sim_config']['recs_per_sample']
        n_per_epoch = self.config['sim_config']['idx_valid'] // rec_idx
        n_tests = float(n_per_epoch // n_per_sample)
        out, sup = self.get_out_and_sup_signals()
        accuracy = []
        for se in range(0, out.shape[0], n_per_epoch):
            ee = se + n_per_epoch
            correct = 0
            for si in range(se, ee, n_per_sample):
                ei = si + n_per_sample
                # m_out = stats.mode(out[si+1:ei], axis=0)[0]
                # m_sup = stats.mode(sup[si+1:ei], axis=0)[0]
                m_out = np.mean(out[si+1:ei], axis=0)
                m_sup = np.max(sup[si+1:ei], axis=0)
                out_win = np.argmax(m_out)
                sup_win = np.argmax(m_sup)
                if out_win == sup_win:
                    correct += 1
            accuracy.append(100. * correct/n_tests)

        return accuracy

    def plot_accuracy(self):
        acc = np.array(self.get_accuracy_signal())
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(acc, alpha=self.alpha, linewidth=self.lw)
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Epoch')
        ax.grid()
        plt.savefig(self.base_dir + "accuracy.png", dpi=300)
        plt.close(fig)
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.plot(100. - acc, alpha=self.alpha, linewidth=self.lw)
        ax.set_ylabel('Error (%)')
        ax.set_xlabel('Epoch')
        ax.grid()
        plt.savefig(self.base_dir + "classification_error.png", dpi=300)
        plt.close(fig)
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.semilogy(100. - acc, alpha=self.alpha, linewidth=self.lw)
        ax.set_ylabel('Error (%)')
        ax.set_xlabel('Epoch')
        ax.grid()
        plt.savefig(self.base_dir + "classification_error_log.png", dpi=300)
        plt.close(fig)

    def plot_signal(self, vals, title, filename=None, epoch_divs=None, logy=False,
                    max_epoch_divs=21, alpha=0.5, label=None, vmin=None, vmax=None,
                    colors=BASIC_COLORS, grid=False, marker=None, window_size=1):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)

        if epoch_divs is not None and len(epoch_divs) < max_epoch_divs:
            for eidx in epoch_divs:
                ax.axvline(eidx, linestyle='--', linewidth=1, color='gray', alpha=0.4)
        if logy:
            ax.semilogy(vals, linewidth=1, alpha=self.alpha)
        else:
            avals = np.asarray(vals)
            for i, v in enumerate(avals.T):
                color_idx = i % len(colors)
                lbl = None if label is None else "{} {}".format(label, i)
                ax.plot(v, linewidth=self.lw, alpha=self.alpha, label=lbl,
                        color=colors[color_idx], marker=marker)
            del avals

            if vmin is not None and vmax is not None:
                ax.set_ylim(vmin, vmax)

        ax.set_title(title)
        ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))

        if grid:
            ax.grid()

        if filename is not None:
            plt.savefig(self.base_dir + filename, dpi=300)

        return fig, ax

    def plot_signals(self, stage, vars=('g', 'V', 'v_apical', 'rate'),
                     vmin=None, vmax=None):
        for lyr in self.paths[stage]:
            for pop in self.paths[stage][lyr]:
                for sig, sig_path in self.paths[stage][lyr][pop].items():
                    # sig = path.basename(path.normpath(sig_path))
                    if sig not in vars:
                        continue
                    elif sig == 'noise':
                        self.analyze_noise(sig_path)
                    elif sig == 'rate_highpass':
                        self.analyze_highpass(sig_path, stage, lyr, pop)

                    print("Plotting {}".format(sig_path))

                    vs = self.get_full_signal(sig_path)
                    title = '{}: full'.format(sig_path)
                    fname = '{}_full.png'.format(
                                sig_path.replace('/', '_'))

                    epoch_ids = None
                    marker = '.'

                    if sig == 'g':
                        marker = None
                    if stage == Stages.VALIDATION:
                        epoch_ids = self._valid_epoch_indices
                    elif stage == Stages.TRAINING:
                        epoch_ids = self._train_epoch_indices

                    n_points = vs.shape[0] * vs.shape[1]

                    window_size = 1
                    if n_points > self.MAX_PLOT_POINTS:
                        n_windows, window_size = self._get_plot_partitions(vs.shape[0], vs.shape[1])

                        [_vs] = self.__process_signals([vs], n_points, n_windows, window_size)
                        del vs
                        vs = np.asarray(_vs)
                        del _vs

                    if sig == 'g' and pop == 'fb':
                        print('saving backw weights for kevin', vs.shape)
                        print(fname)
                        np.save(self.base_dir + fname, vs)
                    fig, ax = self.plot_signal(vs, title, fname, epoch_ids, vmin=vmin,
                            vmax=vmax, marker=None, window_size=window_size)
                    plt.close(fig)
                    del vs
                    # plt.show()

    def plot_signals_epoch(self, stage, epoch, vars=('g', 'V', 'v_apical', 'rate'),
                           vmin=None, vmax=None, check_v_dend=False, check_w_ip=False):
        try:
            dirname = self.base_dir + 'epoch_{}/'.format(epoch)
            os.makedirs(dirname)
            print("Directory ", dirname, " Created ")
        except FileExistsError:
            print("Directory ", dirname, " already exists")

        if stage == Stages.TESTING:
            print('No epoch wise plotting necessary for testing which only runs at the end')
            return
        elif stage == Stages.VALIDATION:
            n_per_epoch = self._valid_epoch_indices[1] # first is 0, second n_per_epoch
            start = n_per_epoch * epoch
            end = n_per_epoch * (epoch + 1)
            if check_v_dend:
                if epoch == 0:
                    w_end = 0
                else:
                    w_n_per_epoch = self._train_epoch_indices[1] # first is 0, second n_per_epoch
                    w_end = w_n_per_epoch * epoch - 1
                sig_path = self.paths[Stages.VALIDATION]['1']['inter']['v_dendrite']
                vs = self.get_full_signal(sig_path)
                v_dend = vs[start:end, :]
                n_inter = v_dend.shape[1]
                sig_path = self.paths[Stages.VALIDATION]['1']['pyramidal']['rate']
                vs = self.get_full_signal(sig_path)
                r_pyr = vs[start:end, :]
                n_pyr = r_pyr.shape[1]
                sig_path = self.paths[Stages.TRAINING]['1']['p2i']['g']
                vs = self.get_full_signal(sig_path)
                w_p2i = vs[w_end, :].reshape(n_pyr, n_inter)
                v_dend_calc = np.matmul(r_pyr, w_p2i)
                for i in range(v_dend_calc.shape[1]):
                    title = 'V_dend_calc vs v_dend, interneuron {0}: epoch {1}'.format(i, epoch)
                    fname = 'epoch_{1}/check_v_dend_inter_{0}_epoch_{1}.png'.format(i, epoch)
                    fig = plt.figure()
                    ax = plt.subplot(1, 1, 1)
                    ax.plot(v_dend_calc[:, i], label='calc', alpha=0.6)
                    ax.plot(v_dend[:, i], label='sim', alpha=0.6)
                    ax.legend()
                    plt.savefig(self.base_dir + fname)
                    plt.close(fig)
        else:
            if len(self._train_epoch_indices) == 1:
                start = 0
                rec_idx = self.config['sim_config']['idx_rec']
                end = self.config['sim_config']['idx_train'] // rec_idx
            else:
                n_per_epoch = self._train_epoch_indices[1] # first is 0, second n_per_epoch
                start = n_per_epoch * (epoch - 1)
                end = n_per_epoch * epoch
            if epoch == 0:
                print('The epochs are counted starting from one to accomodate for the\
                       validation run before training start (that one is plotted if stage\
                       is validation and the epoch is 0. To plot results after first training epoch, use epoch = 1.')
                return
            if check_w_ip:
                sig_path = self.paths[Stages.VALIDATION]['1']['inter']['rate_dendrite']
                vs = self.get_full_signal(sig_path)
                r_bas_inter = vs[start:end, :]
                sig_path = self.paths[Stages.VALIDATION]['1']['inter']['rate']
                vs = self.get_full_signal(sig_path)
                r_inter = vs[start:end, :]
                n_inter = r_inter.shape[1]
                sig_path = self.paths[Stages.VALIDATION]['1']['pyramidal']['rate']
                vs = self.get_full_signal(sig_path)
                r_pyr = vs[start:end, :]
                n_pyr = r_pyr.shape[1]
                sig_path = self.paths[Stages.TRAINING]['1']['p2i']['g']
                vs = self.get_full_signal(sig_path)
                w_p2i = vs[start:end, :]
                n_steps = r_pyr.shape[0]
                for i in range(n_steps):
                    dw = w_p2i[i] - w_p2i[i-1]
                    dw = dw.reshape(n_pyr, n_inter)
                    print(dw.shape)
                    print('sim dw', dw)
                    r_diff = r_inter[i] - r_bas_inter[i]
                    r_diff = r_diff.reshape(-1, 1)
                    print(r_diff.shape)
                    print('r_diff', r_diff)
                    r_p = r_pyr[i].reshape(1, -1)
                    print(r_p.shape)
                    print('r_pyr', r_p)
                    dw_calc = 0.032 * np.matmul(r_diff, r_p)
                    print('dw_calc', dw_calc)
                    print('#############')

        for lyr in self.paths[stage]:
            for pop in self.paths[stage][lyr]:
                for sig, sig_path in self.paths[stage][lyr][pop].items():
                    # sig = path.basename(path.normpath(sig_path))
                    if sig not in vars:
                        continue

                    print("Plotting {}".format(sig_path))

                    vs = self.get_full_signal(sig_path)
                    vs = vs[start:end, :]
                    title = '{0}: epoch {1}'.format(sig_path, epoch)
                    fname = 'epoch_{1}/{0}_epoch_{1}.png'.format(
                                sig_path.replace('/', '_'), epoch)

                    epoch_ids = None
                    marker = '.'

                    if sig == 'g':
                        marker = None
                    if stage == Stages.VALIDATION:
                        epoch_ids = self._valid_epoch_indices
                    elif stage == Stages.TRAINING:
                        epoch_ids = self._train_epoch_indices

                    n_points = vs.shape[0] * vs.shape[1]

                    window_size = 1
                    if n_points > self.MAX_PLOT_POINTS:
                        n_windows, window_size = self._get_plot_partitions(vs.shape[0], vs.shape[1])

                        [_vs] = self.__process_signals([vs], n_points, n_windows, window_size)
                        del vs
                        vs = np.asarray(_vs)
                        del _vs

                    fig, ax = self.plot_signal(vs, title, fname, epoch_ids, vmin=vmin,
                            vmax=vmax, marker=None, window_size=window_size)
                    plt.close(fig)
                    del vs

    def analyze_noise(self, sig_path):
        vs = self.get_full_signal(sig_path)
        print(vs.shape)
        print('mean:', np.mean(vs, axis=0))
        print('stddev:', np.std(vs, axis=0))
        from statsmodels.tsa.stattools import acf
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        ax.set_title('Autocorrelation of OU noise')
        for nrn in range(vs.shape[1]):
            to_plot = acf(vs[:, nrn])
            ax.plot(to_plot, alpha=0.7)

        ax.set_xlabel('time steps')
        plt.savefig(self.base_dir + 'autokov_noise_pyr.png', dpi=300)
        plt.close(fig)
        del vs

    def analyze_highpass(self, sig_path, stage, lyr, pop):
        rate_path = self.paths[stage][lyr][pop]['rate']
        highpassed = self.get_full_signal(sig_path)
        rates = self.get_full_signal(rate_path)
        fname = '{}_minus_highpass_full.png'.format(rate_path.replace('/', '_'))
        fig = plt.figure()
        title = '{0} - highpassed'.format(rate_path)
        ax = plt.subplot(1, 1, 1)
        ax.set_title('Pyramidal rates vs highpassed rates')
        for nrn in range(highpassed.shape[1]):
            ax.plot(rates[:, nrn], alpha=0.5, label='r', color='C{}'.format(nrn))
            ax.plot(rates[:, nrn] - highpassed[:, nrn], alpha=0.7, label='r - r_hat', ls='--', color='C{}'.format(nrn))
        ax.set_xlabel('time steps')
        ax.legend()
        plt.savefig(self.base_dir + fname, dpi=300)
        plt.close(fig)
        del rates
        del highpassed

    def plot_angle_ff_fb(self, network_layout, return_angle=False):
        stage = 'training'
        print('starting angle plot', flush=True)
        print(return_angle, flush=True)

        def cos_sim(A, B):
            if A.ndim == 1 and B.ndim == 1:
                return A.T @ B / np.linalg.norm(A) / np.linalg.norm(B)
            else:
                return np.trace(A.T @ B) / np.linalg.norm(A) / np.linalg.norm(B)

        if return_angle:
            return_dict = {}
        for i, layer_id in enumerate(self._sorted_layer_ids):
            if int(layer_id) == 0:
                continue
            if layer_id == self.output_layer_id:
                break

            next_id = self._sorted_layer_ids[i+1]
            ff, fb = self.get_ff_and_fb(stage, layer_id, next_id)
            n_points = ff.shape[0] * ff.shape[1]
            window_size = 1
            if n_points > self.MAX_PLOT_POINTS:
                n_windows, window_size = self._get_plot_partitions(ff.shape[0], ff.shape[1])

                [ffn, fbn] = self.__process_signals([ff, fb],
                                                n_points, n_windows, window_size)

                del ff
                del fb
                ff = np.asarray(ffn)
                fb = np.asarray(fbn)
                del ffn
                del fbn
            else:
                ff = np.asarray(ff)
                fb = np.asarray(fb)

            n_pyr_next = network_layout[int(next_id)]
            n_pyr = network_layout[int(layer_id)]
            ff = ff.reshape(ff.shape[0], n_pyr, n_pyr_next)
            fb = fb.reshape(fb.shape[0], n_pyr_next, n_pyr)
            fb = fb.transpose(0, 2, 1)
            angles = []
            # TODO: deuglyfy
            for j in range(len(ff)):
                cos = cos_sim(ff[j], fb[j])
                alpha = np.arccos(cos) * 180 / np.pi
                angles.append(alpha)

            if return_angle:
                return_dict[layer_id] = angles
            else:
                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("angle between ff and fb from layer {}".format(i))
                plt.plot(angles, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_angle_ff_fb.png".format(i, stage), dpi=300)
                plt.close(fig)

            del ff, fb

        print("END: plot_angle_ff_fb", flush=True)
        if return_angle:
            return return_dict


    def plot_error_signal(self):
        mse = self.get_error_signal()
        fig, ax = self.plot_signal(mse, 'Mean Square Error validation',
                                   'mse_validation.png', logy=True)
        plt.close(fig)
        del mse
        print("END: plot_error_signal")

    def plot_out_vs_sup(self, splits_per_epoch=1, vmax=None, vmin=None):
        out0, sup0 = self.get_out_and_sup_signals()
        out_test, sup_test = self.get_out_and_sup_signals(which_stage='testing')

        out = out_test[:, :]
        sup = sup_test[:, :]
        fig, ax = self.plot_signal(sup, 'Output vs. Supervisor (test set)', 
                                   label='Supervisor')
        for j, v in enumerate(out.T):
            ax.plot(v, label='Output {}'.format(j),
                    linestyle='--', alpha=self.alpha, linewidth=self.lw)
        plt.legend()
        plt.savefig(self.base_dir + 'output_vs_supervisor_testing.png', dpi=300)
        plt.close(fig)

        # plot just last epoch
        last_epoch_idx = self._valid_epoch_indices[-1]
        n_per_epoch = self._valid_epoch_indices[1] # first is 0, second n_per_epoch
        # out, sup = self.get_out_and_sup_signals()
        out = out0[last_epoch_idx:, :]
        sup = sup0[last_epoch_idx:, :]

        fig, ax = self.plot_signal(sup, 'Output vs. Supervisor (last epoch)', 
                                   label='Supervisor')
        for j, v in enumerate(out.T):
            ax.plot(v, label='Output {}'.format(j),
                    linestyle='--', alpha=self.alpha, linewidth=self.lw)
        plt.legend()
        plt.savefig(self.base_dir + 'output_vs_supervisor_validation_last_epoch.png', dpi=300)
        plt.close(fig)

        for i, se in enumerate(self._valid_epoch_indices):
            ee = (None if (i + 1) == len(self._valid_epoch_indices) else
                  self._valid_epoch_indices[i + 1])
            out = out0[se:ee, :]
            sup = sup0[se:ee, :]

            n_per_split = n_per_epoch//splits_per_epoch
            for j, split_start in enumerate(range(0, n_per_epoch, n_per_split)):
                title = 'Output vs. Supervisor (epoch {}, split {})'.format(i, j)
                split_end = min(n_per_epoch, split_start + n_per_split)
                ssup = sup[split_start:split_end]
                sout = out[split_start:split_end]
                fig, ax = self.plot_signal(ssup, title, label='Supervisor')
                for k, v in enumerate(sout.T):
                    ax.plot(v, label='Output {}'.format(k),
                            linestyle='--', alpha=self.alpha, linewidth=self.lw,
                            color=BASIC_COLORS[k])
                plt.legend()
                fname = 'output_vs_supervisor_validation_{:05d}_{:05d}.png'.format(i, j)
                if vmax is not None and vmin is not None:
                    ax.set_ylim(vmin, vmax)
                plt.savefig(self.base_dir + fname, dpi=300)
                plt.close(fig)

        del out0, sup0
        print("END: plot_out_vs_sup")

    def get_ff_and_p2i(self, stage, lyr, next):
        ff_path = self.paths[stage][str(next)]['ff']['g']
        p2i_path = self.paths[stage][str(lyr)]['p2i']['g']
        return (self.get_full_signal(ff_path),
                self.get_full_signal(p2i_path))

    def get_fb_and_i2p(self, stage, lyr):
        fb_path = self.paths[stage][str(lyr)]['fb']['g']
        i2p_path = self.paths[stage][str(lyr)]['i2p']['g']
        return (self.get_full_signal(fb_path),
                self.get_full_signal(i2p_path))

    def get_ff_and_fb(self, stage, lyr, next):
        ff_path = self.paths[stage][str(next)]['ff']['g']
        fb_path = self.paths[stage][str(lyr)]['fb']['g']
        return (self.get_full_signal(ff_path),
                self.get_full_signal(fb_path))

    @staticmethod
    # @jit(nopython=True)
    def __process_signals(signals, n_points, n_windows, window_size):
        n_signals = len(signals)
        s = [[] for _ in signals]
        for j in range(n_windows):
            w0 = j * window_size
            w1 = w0 + window_size
            if w1 > n_points:
                w1 = n_points
            for k in range(len(signals)):
                skj = signals[k][w0:w1, :]
                s[k].append(np.mean(skj, axis=0))
                if np.any(np.isnan(np.mean(skj, axis=0))):
                    print(j, k)
                    print(skj.shape)
                    sys.exit()
        return s

    def _get_plot_partitions(self, n_time_steps, n_curves):
        window_size = int(np.ceil(n_time_steps / (self.MAX_PLOT_POINTS / n_curves)))
        if window_size > 0:
            n_windows = max(1, n_time_steps // window_size)
        else:
            n_windows = 1
        return n_windows, window_size

    def plot_next_ff_vs_p2i(self):

        for si, stage in enumerate(self._stages):
            for i, layer_id in enumerate(self._sorted_layer_ids):
                if int(layer_id) == 0:
                    continue
                if layer_id == self.output_layer_id:
                    break

                next_id = self._sorted_layer_ids[i+1]
                ff, p2i = self.get_ff_and_p2i(stage, layer_id, next_id)
                n_points = ff.shape[0] * ff.shape[1]
                window_size = 1
                if n_points > self.MAX_PLOT_POINTS:
                    n_windows, window_size = self._get_plot_partitions(ff.shape[0], ff.shape[1])

                    [ffn, p2in] = self.__process_signals([ff, p2i],
                                                    n_points, n_windows, window_size)

                    del ff
                    del p2i
                    ff = np.asarray(ffn)
                    p2i = np.asarray(p2in)
                    del ffn
                    del p2in
                else:
                    ff = np.asarray(ff)
                    p2i = np.asarray(p2i)

                diff = ff - p2i
                abs_diff = np.abs(diff)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("log abs weight diff from layer {} out_ff vs p2i".format(i))
                plt.semilogy(abs_diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_ff_vs_p2i_log.png".format(i, stage), dpi=300)
                plt.close(fig)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("abs weight diff from layer {} out_ff vs p2i".format(i))
                plt.plot(abs_diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_ff_vs_p2i.png".format(i, stage), dpi=300)
                plt.close(fig)

                del ff, p2i, diff, abs_diff

        print("END: plot_next_ff_vs_i2p")

    def plot_next_ff_vs_p2i_epoch(self, epoch):
        try:
            dirname = self.base_dir + 'epoch_{}/'.format(epoch)
            os.makedirs(dirname)
            print("Directory ", dirname, " Created ")
        except FileExistsError:
            print("Directory ", dirname, " already exists")

        stage = Stages.TRAINING # no weight changes in validation or testing
        if len(self._train_epoch_indices) == 1:
            start = 0
            rec_idx = self.config['sim_config']['idx_rec']
            end = self.config['sim_config']['idx_train'] // rec_idx
        else:
            n_per_epoch = self._train_epoch_indices[1] # first is 0, second n_per_epoch
            start = n_per_epoch * (epoch - 1)
            end = n_per_epoch * epoch
        if epoch == 0:
            print('The epochs are counted starting from one to accomodate for the\
                   validation run before training start (that one is plotted if stage\
                   is validation and the epoch is 0. To plot results after first training epoch, use epoch = 1.')

        for i, layer_id in enumerate(self._sorted_layer_ids):
            if int(layer_id) == 0:
                continue
            if layer_id == self.output_layer_id:
                break

            next_id = self._sorted_layer_ids[i+1]
            ff, p2i = self.get_ff_and_p2i(stage, layer_id, next_id)
            ff = ff[start:end, :]
            p2i = p2i[start:end, :]
            n_points = ff.shape[0] * ff.shape[1]
            window_size = 1
            if n_points > self.MAX_PLOT_POINTS:
                n_windows, window_size = self._get_plot_partitions(ff.shape[0], ff.shape[1])

                [ffn, p2in] = self.__process_signals([ff, p2i],
                                                n_points, n_windows, window_size)

                del ff
                del p2i
                ff = np.asarray(ffn)
                p2i = np.asarray(p2in)
                del ffn
                del p2in
            else:
                ff = np.asarray(ff)
                p2i = np.asarray(p2i)

            diff = ff - p2i
            abs_diff = np.abs(diff)

            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
            ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
            ax.set_title("log abs weight diff from layer {} out_ff vs p2i".format(i))
            plt.semilogy(abs_diff, alpha=self.alpha, linewidth=self.lw)
            ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
            plt.savefig(self.base_dir + "epoch_{2}/{0:03d}_{1}_ff_vs_p2i_log_epoch_{2}.png".format(i, stage, epoch), dpi=300)
            plt.close(fig)

            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
            ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
            ax.set_title("abs weight diff from layer {} out_ff vs p2i".format(i))
            plt.plot(abs_diff, alpha=self.alpha, linewidth=self.lw)
            ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
            plt.savefig(self.base_dir + "epoch_{2}/{0:03d}_{1}_ff_vs_p2i_epoch_{2}.png".format(i, stage, epoch), dpi=300)
            plt.close(fig)

            del ff, p2i, diff, abs_diff

        print("END: plot_next_ff_vs_i2p")

    def plot_next_fb_vs_i2p(self):

        for si, stage in enumerate(self._stages):
            for i, layer_id in enumerate(self._sorted_layer_ids):
                if int(layer_id) == 0:
                    continue
                if layer_id == self.output_layer_id:
                    break

                next_id = self._sorted_layer_ids[i+1]
                fb, i2p = self.get_fb_and_i2p(stage, layer_id)
                n_points = fb.shape[0] * fb.shape[1]
                window_size = 1
                if n_points > self.MAX_PLOT_POINTS:
                    n_windows, window_size = self._get_plot_partitions(fb.shape[0], fb.shape[1])

                    [fbn, i2pn] = self.__process_signals([fb, i2p],
                                                    n_points, n_windows, window_size)

                    del fb
                    del i2p
                    fb = np.asarray(fbn)
                    i2p = np.asarray(i2pn)
                    del fbn
                    del i2pn
                else:
                    fb = np.asarray(fb)
                    i2p = np.asarray(i2p)

                diff = fb + i2p
                abs_diff = np.abs(diff)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("log abs weight diff from layer {} out_ff vs p2i".format(i))
                plt.semilogy(abs_diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_fb_vs_i2p_log.png".format(i, stage), dpi=300)
                plt.close(fig)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("abs weight diff from layer {} out_ff vs p2i".format(i))
                plt.plot(abs_diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_fb_vs_i2p.png".format(i, stage), dpi=300)
                plt.close(fig)

                del fb, i2p, diff, abs_diff

        print("END: plot_next_fb_vs_p2i")

    def get_next_and_inter_rates(self, stage, lyr, next, is_next_output):
        cell = 'output' if is_next_output else 'pyramidal'
        next_path = self.paths[stage][next][cell]['rate']
        inter_path = self.paths[stage][lyr]['inter']['rate']
        return (self.get_full_signal(next_path),
                self.get_full_signal(inter_path))

    def plot_next_vs_inter_rate(self):
        for si, stage in enumerate(self._stages):
            for i, layer_id in enumerate(self._sorted_layer_ids):
                if int(layer_id) == 0:
                    continue
                if layer_id == self.output_layer_id:
                    break

                next_id = self._sorted_layer_ids[i + 1]
                is_next_output = next_id == self.output_layer_id
                _next, _inter = self.get_next_and_inter_rates(
                                    stage, layer_id, next_id, is_next_output)
                n_points = _next.shape[0] * _next.shape[1]
                window_size = 1
                if n_points > self.MAX_PLOT_POINTS:
                    n_windows, window_size = self._get_plot_partitions(_next.shape[0], _next.shape[1])

                    [_next_n, _inter_n] = self.__process_signals([_next, _inter],
                                                    n_points, n_windows, window_size)
                    del _next
                    del _inter
                    _next = np.asarray(_next_n)
                    _inter = np.asarray(_inter_n)
                    del _next_n
                    del _inter_n
                else:
                    _next = np.asarray(_next)
                    _inter = np.asarray(_inter)

                diff = _next - _inter
                abs_diff = np.abs(diff)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("log abs(rate diff) from layer {} next vs inter".format(i))
                plt.semilogy(abs_diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_next_vs_inter_rate_log.png".format(i, stage), dpi=300)
                plt.close(fig)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("rate diff from layer {} next vs inter".format(i))
                plt.plot(diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "{0:03d}_{1}_next_vs_inter_rate.png".format(i, stage), dpi=300)
                plt.close(fig)

                del _next, _inter, diff, abs_diff

        print("END: plot_next_vs_inter")


    def plot_next_vs_inter_rate_epoch(self, epoch):
        try:
            dirname = self.base_dir + 'epoch_{}/'.format(epoch)
            os.makedirs(dirname)
            print("Directory ", dirname, " Created ")
        except FileExistsError:
            print("Directory ", dirname, " already exists")

        for stage in [Stages.TRAINING, Stages.VALIDATION]:
            if stage == Stages.VALIDATION:
                n_per_epoch = self._valid_epoch_indices[1] # first is 0, second n_per_epoch
                start = n_per_epoch * epoch
                end = n_per_epoch * (epoch + 1)
            else:
                if len(self._train_epoch_indices) == 1:
                    start = 0
                    rec_idx = self.config['sim_config']['idx_rec']
                    end = self.config['sim_config']['idx_train'] // rec_idx
                else:
                    n_per_epoch = self._train_epoch_indices[1] # first is 0, second n_per_epoch
                    start = n_per_epoch * (epoch - 1)
                    end = n_per_epoch * epoch
                if epoch == 0:
                    print('The epochs are counted starting from one to accomodate for the\
                           validation run before training start (that one is plotted if stage\
                           is validation and the epoch is 0. To plot results after first training epoch, use epoch = 1.')
                    continue

            for i, layer_id in enumerate(self._sorted_layer_ids):
                if int(layer_id) == 0:
                    continue
                if layer_id == self.output_layer_id:
                    break

                next_id = self._sorted_layer_ids[i + 1]
                is_next_output = next_id == self.output_layer_id
                _next, _inter = self.get_next_and_inter_rates(
                                    stage, layer_id, next_id, is_next_output)
                _next = _next[start:end, :]
                _inter = _inter[start:end, :]
                n_points = _next.shape[0] * _next.shape[1]
                window_size = 1
                if n_points > self.MAX_PLOT_POINTS:
                    n_windows, window_size = self._get_plot_partitions(_next.shape[0], _next.shape[1])

                    [_next_n, _inter_n] = self.__process_signals([_next, _inter],
                                                    n_points, n_windows, window_size)
                    del _next
                    del _inter
                    _next = np.asarray(_next_n)
                    _inter = np.asarray(_inter_n)
                    del _next_n
                    del _inter_n
                else:
                    _next = np.asarray(_next)
                    _inter = np.asarray(_inter)

                diff = _next - _inter
                abs_diff = np.abs(diff)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("log abs(rate diff) from layer {} next vs inter".format(i))
                plt.semilogy(abs_diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "epoch_{2}/{0:03d}_{1}_next_vs_inter_rate_log_epoch_{2}.png".format(i, stage, epoch), dpi=300)
                plt.close(fig)

                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axhline(0, linestyle=':', color='gray', linewidth=0.5)
                ax.set_title("rate diff from layer {} next vs inter".format(i))
                plt.plot(diff, alpha=self.alpha, linewidth=self.lw)
                ax.set_xlabel('accumulation windows (averaged over {} rec steps)'.format(window_size))
                plt.savefig(self.base_dir + "epoch_{2}/{0:03d}_{1}_next_vs_inter_rate_epoch_{2}.png".format(i, stage, epoch), dpi=300)
                plt.close(fig)

                del _next, _inter, diff, abs_diff

        print("END: plot_next_vs_inter")

    def plot_weights(self, vmin=None, vmax=None, splits_per_epoch=1):
        n_epochs = self.config['sim_config']['n_epochs']
        inds = self.config['record']['update_indices']
        n_epoch_steps = inds['train'] + inds['valid']
        n_epochs_per_write = float(inds['write']) / n_epoch_steps
        n_weight_recs = int((n_epochs_per_write * inds['train']) // inds['weights'])
        recs_per_epoch = int(n_weight_recs // n_epochs)
        epoch_starts = np.arange(0, n_weight_recs, recs_per_epoch)
        n_per_split = recs_per_epoch // splits_per_epoch
        split_starts = np.arange(0, recs_per_epoch, n_per_split)
        filename_template = "weights_layer_{:03d}_pop_{}_epoch_{:03d}_div_{:03d}.png"

        for lyr in self.paths:
            for pop in self.paths[lyr]:
                for sig, sig_path in self.paths[lyr][pop].items():
                    if sig != 'g':
                        continue

                    vs = self.get_full_signal(sig_path)
                    if vmax is None:
                        maxs = []
                        for epoch_idx, epoch_start in enumerate(epoch_starts):
                            for split_idx, split_start in enumerate(
                                    split_starts):
                                s = epoch_start + split_start
                                e = min(epoch_start + recs_per_epoch,
                                        s + n_per_split)
                                maxs.append(np.max(np.abs(vs[s:e, :])))
                        _vmax = np.max(maxs)
                        _vmin = -_vmax
                        del maxs
                    else:
                        _vmax = vmax
                        _vmin = vmin

                        # vmax = np.max(np.abs(vs)) if vmax is None else vmax
                        # vmin = -vmax if vmin is None else vmin

                    for epoch_idx, epoch_start in enumerate(epoch_starts):
                        for split_idx, split_start in enumerate(split_starts):
                            s = epoch_start + split_start
                            e = min(epoch_start + recs_per_epoch, s + n_per_split)

                            fig = plt.figure()
                            ax = plt.subplot(1, 1, 1)
                            ax.set_title("lyr {} w {}:   e {}/{} s {}/{}".format(
                                lyr, pop,
                                epoch_idx, epoch_starts.size,
                                split_idx, split_starts.size))
                            ax.plot(np.arange(s, e), vs[s:e, :],
                                    linewidth=self.linewidth, alpha=self.alpha)
                            ax.set_ylim(_vmin, _vmax)
                            fname = filename_template.format(
                                        int(lyr), pop, epoch_idx, split_idx)
                            plt.savefig(self.base_dir + fname, dpi=300)
                            plt.close(fig)

    def _get_validation_signal_metric_per_epoch(self, signal, epoch_id, mode_axis=0,
                                                metric:str='mean'):
        '''metric can be mode, median, mean, max, min'''
        cfg = self.config
        rec_idx = cfg['sim_config']['idx_rec']
        recs_per_sample = cfg['sim_config']['recs_per_sample']
        n_per_validation = cfg['sim_config']['idx_valid']
        recs_per_epoch =  n_per_validation // rec_idx
        mlow = metric.lower()
        ins = []
        epoch_starts = np.arange(0, signal.shape[0], recs_per_epoch)
        for iepoch, se in enumerate(epoch_starts):
            ee = se + recs_per_epoch
            sample_starts = np.arange(se, ee, recs_per_sample)
            if iepoch != epoch_id:
                continue

            for si in sample_starts:
                ei = si + recs_per_sample
                # sig0 = signal[si:ei]
                sig = signal[si+1:ei]
                if mlow == 'mode':
                    m = stats.mode(sig, axis=mode_axis)[0][0]
                elif mlow == 'mean':
                    m = np.mean(sig, axis=mode_axis)
                elif mlow == 'median':
                    m = np.median(sig, axis=mode_axis)
                elif mlow == 'max':
                    m = np.max(sig, axis=mode_axis)
                elif mlow == 'min':
                    m = np.min(sig, axis=mode_axis)
                else:
                    raise Exception("metric {} is not implemented "
                                    "or invalid ".format(mlow))

                ins.append(m)
        return ins

    def get_validation_inputs_per_epoch(self, epoch_id):
        lyr = '0'
        if not ('input' in self.paths[Stages.VALIDATION][lyr] or
                'rate' in self.paths[Stages.VALIDATION][lyr]['input']):
            warn("No rate or input signal found in the record")
            return []
        path = self.paths[Stages.VALIDATION][lyr]['input']['rate']
        sig = self.get_signal_for_validation(path)
        met = self._get_validation_signal_metric_per_epoch(sig, epoch_id)
        amet = np.asarray(met)
        self._differences_input[epoch_id] = amet
        return met

    def get_validation_outputs_per_epoch(self, epoch_id):
        lyr = self.output_layer_id
        if not ('output' in self.paths[Stages.VALIDATION][lyr] or
                'V' in self.paths[Stages.VALIDATION][lyr]['output']):
            warn("No V or output signal found in the record")
            return []

        path = self.paths[Stages.VALIDATION][lyr]['output']['v_brev']
        sig = self.get_signal_for_validation(path)
        return self._get_validation_signal_metric_per_epoch(sig, epoch_id)

    def get_validation_supervision_per_epoch(self, epoch_id):
        lyr = self.supervisor_layer_id
        if not ('supervisor' in self.paths[Stages.VALIDATION][lyr] or
                'rate' in self.paths[Stages.VALIDATION][lyr]['output']):
            warn("No rate or supervisor signal found in the record")
            return []

        path = self.paths[Stages.VALIDATION][lyr]['supervisor']['rate']
        sig = self.get_signal_for_validation(path)
        return self._get_validation_signal_metric_per_epoch(
                                                    sig, epoch_id, metric='max')

    def get_n_recs_per_epoch(self, stage, rec_type):
        cfg = self.config
        rec_idx = cfg['sim_config']['idx_rec']
        # since recordings are now split, each stage has its own 'number of steps'
        k = 'idx_train' if (rec_type == 'weights' or stage == Stages.TRAINING) else 'idx_valid'
        steps_per_epoch = cfg['sim_config'][k]
        return (steps_per_epoch // rec_idx)

    def plot_inputs(self, stage, start_epoch=0, splits_per_epoch=1):
        inp = self.get_full_signal(os.path.join(stage, '0/input/rate'))
        cfg = self.config
        recs_per_epoch = self.get_n_recs_per_epoch(stage, 'rate')
        n_epochs = cfg['sim_config']['n_epochs']
        total_recs = n_epochs * recs_per_epoch
        n_per_split = recs_per_epoch // splits_per_epoch
        for ei, es in enumerate(np.arange(0, total_recs, recs_per_epoch)):
            if ei < start_epoch:
                continue
            ee = es + recs_per_epoch
            for si, ss in enumerate(np.arange(es, ee, n_per_split)):
                sys.stdout.write(
                    "\rInput stage {}\tepoch {:05d}\tsplit {:05d}".format(stage, ei, si))
                sys.stdout.flush()

                se = min(total_recs, ss + n_per_split)
                sig = inp[ss: se, :2]
                title = "input stage {} epoch {:05d}  split {:05d}".format(stage, ei, si)
                fname = "{}.png".format(title.replace(" ", "_"))
                fig, ax = self.plot_signal(sig, title, fname, grid=True)
                plt.close(fig)

    # Only for the yin yang dataset
    def plot_yy_points(self):
        if self.config['dataset_key'] == 'yinyang':
            _colors = ['red', 'green', 'blue']
            n_epochs = self.config['sim_config']['n_epochs']
            for epoch in range(n_epochs):
                in_rates = self.get_validation_inputs_per_epoch(epoch)
                out_rates = self.get_validation_outputs_per_epoch(epoch)
                sup_rates = self.get_validation_supervision_per_epoch(epoch)

                first_class = [True for _ in _colors]
                fig = plt.figure()
                ax = plt.subplot(1, 1, 1)
                ax.axis('equal')
                total = float(len(sup_rates))
                correct = 0.0
                for sidx in range(len(in_rates)):
                    sys.stdout.write(
                        "\rProcessing epoch {:3d}/{:3d} inputs {:4d}/{:4d}".format(
                        epoch + 1, n_epochs, sidx + 1, len(in_rates)
                    ))
                    sys.stdout.flush()
                    x, y = in_rates[sidx][:2]
                    c = int(np.argmax(out_rates[sidx]))
                    s = int(np.argmax(sup_rates[sidx]))
                    # print(x, y, c, s)
                    if c == s:
                        correct += 1.0

                    label = c if first_class[c] else None
                    first_class[c] = False
                    ax.plot(x, y, markerfacecolor='none', marker='o',
                            markeredgewidth=1., markeredgecolor=_colors[c],
                            linestyle='none',
                            label=label)
                    ax.set_aspect('equal', 'box')
                    ax.set_xlim(-0.2, 1.2)
                    ax.set_ylim(-0.2, 1.2)

                ax.set_title("Epoch {:03d} acc {:6.2f}".format(epoch, 100.0 * correct / total))
                ax.legend()

                plt.savefig(self.base_dir + "points_epoch_{:03d}.png".format(epoch), dpi=300)
                plt.close(fig)


if __name__ == '__main__':
    import sys

    fname = sys.argv[1]
    print("Analizing {}".format(fname))
    an = Analyzer(fname, plots_alpha=1.0, linewidth=0.5)
    # epoch = 0
    # in_rates = an.get_validation_inputs_per_epoch(epoch)
    # out_rates = an.get_validation_outputs_per_epoch(epoch)
    an.plot_error_signal()
    an.plot_accuracy()
    for stage in stages:
        an.plot_inputs(stage)#start_epoch=39, splits_per_epoch=200)
    an.plot_out_vs_sup()#50)
    an.plot_next_ff_vs_p2i()
    an.plot_next_vs_inter_rate()
    # vmax = 1
    splits_per_epoch = 1
    for stage in stages:
        an.plot_signals(stage, vars=['g', 'rate', 'v_apical', 'V'])

    an.plot_weights(
        # vmin=-vmax, vmax=vmax,
        splits_per_epoch=splits_per_epoch
    )
    # an.plot_signals(['g'], vmin=-vmax, vmax=vmax)
