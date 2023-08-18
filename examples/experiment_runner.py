import os
import yaml
import time
import copy
import pprint
import datetime
import subprocess
import numpy as np

from microcircuits import Stages
from microcircuits.genn_models import Network
from microcircuits.Dataset import BarsDataset, YinYangDataset
from microcircuits.genn_models import Analyzer

def load_config(dirname, loading_template=False, verbose=False):
    if loading_template:
        path = dirname
    else:
        path = dirname + '/' + 'config.yaml'
    with open(path) as f:
        data = yaml.load_all(f, Loader=yaml.Loader)
        all_configs = next(iter(data))
    meta_params = all_configs[0]
    network_params = all_configs[1]
    if verbose:
        print('meta_params: ', meta_params)
        print('network_params: ', network_params)
    return meta_params, network_params


def save_config(dirname, meta_params, network_params, epoch_dir=(False, -1)):
    if not dirname[-1] == '/':
        dirname += '/'
    if epoch_dir[0]:
        dirname += 'epoch_{}/'.format(epoch_dir[1])
    try:
        os.makedirs(dirname)
        print("Directory ", dirname, " Created ")
    except FileExistsError:
        print("Directory ", dirname, " already exists")
    # save parameter configs
    with open(dirname + 'config.yaml', 'w') as f:
        yaml.dump([meta_params, network_params], f)
    with open(dirname + 'gitsha.txt', 'w') as f:
        f.write(subprocess.check_output(["git", "rev-parse", "HEAD"]).decode())
    return


class Experiment(object):
    def __init__(self, basepath, config_path=None, loading_existing=False):
        self.basepath = basepath
        self.config_path = config_path
        if config_path is None and not loading_existing:
            raise ValueError('Either provide a config_path or load an existing run')
        if loading_existing:
            self.dirname = self.basepath
            self.meta_params, self.network_params = load_config(self.dirname)
        else:
            self.meta_params, self.network_params = load_config(self.config_path, loading_template=True)
            # after loading template, now save used config in experiment dir
            self.dirname = self.basepath + '{0}_{1:%Y-%m-%d_%H-%M-%S}/'.format(self.meta_params['name'], datetime.datetime.now())
            self.fill_out_config_template()
            save_config(self.dirname, self.meta_params, self.network_params)
        self.setup()
        return

    def setup(self, timeit=True):
        t1 = time.time()
        combi_params = self.meta_params.copy()
        combi_params.update(self.network_params)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(combi_params)
        self.net = Network(combi_params)
        t2 = time.time()
        print('Setup took {} sec'.format(t2-t1))
        return

    def run(self, timeit=True):
        t1 = time.time()
        self.net.run(recording_config=self.meta_params['record'],
                never_use_teach_signal=False)
        t2 = time.time()
        print('Simulating took {} sec'.format(t2-t1))
        return

    def fill_out_config_template(self):
        self.load_dataset()
        self.meta_params['record']['filename'] = self.dirname + 'net_recordings.h5'
        pattern_dur = self.meta_params['sim_config']['pat_t']
        dt = self.meta_params['dt']
        recs_per_sample = self.meta_params['sim_config']['recs_per_sample']
        n_train = self.meta_params['sim_config']['n_train']
        n_valid = self.meta_params['sim_config']['n_valid']
        n_test = self.meta_params['sim_config']['n_test']
        n_class = self.meta_params['sim_config']['n_class']
        n_epochs = self.meta_params['sim_config']['n_epochs']
        # calculate number of steps per sample/epoch/training-epoch etc
        idx_sample = int(pattern_dur / dt)
        idx_rec = max(1, idx_sample // recs_per_sample)
        idx_epoch = int((n_train + n_valid) * n_class * (pattern_dur / dt))
        idx_train = int(n_class * n_train * (pattern_dur / dt))
        idx_valid = int(n_class * n_valid * (pattern_dur / dt))
        idx_test = int(n_class * n_test * (pattern_dur / dt))
        idx_write = idx_epoch * 2
        sim_time = (n_train + n_valid) * n_class * n_epochs * pattern_dur
        self.meta_params['sim_config']['idx_sample'] = idx_sample
        self.meta_params['sim_config']['idx_rec'] = idx_rec
        self.meta_params['sim_config']['idx_epoch'] = idx_epoch
        self.meta_params['sim_config']['idx_train'] = idx_train
        self.meta_params['sim_config']['idx_valid'] = idx_valid
        self.meta_params['sim_config']['idx_test'] = idx_test
        self.meta_params['sim_config']['sim_time'] = sim_time
        self.meta_params['record']['update_indices'] = {
            'weights': idx_rec,
            'rates': idx_rec,
            'write': idx_write,
            'train': idx_train,
            'valid': idx_valid,
            'test': idx_test,
            'sample': idx_sample,
        }
        return

    def load_dataset(self):
        if self.meta_params['dataset_key'] == 'bars':
            self.load_bars_dataset()
        elif self.meta_params['dataset_key'] == 'mimic':
            self.load_mimic_dataset()
        elif self.meta_params['dataset_key'] == 'yinyang':
            self.load_yin_yang_dataset()
        elif self.meta_params['dataset_key'] == 'mnist':
            self.load_mnist_dataset()
        else:
            raise NotImplementedError

    def load_bars_dataset(self):
        np.random.seed(self.meta_params['sim_config']['data_seed'])
        train_rates, train_classes = BarsDataset(3, samples_per_class=self.meta_params['sim_config']['n_train'],
                                                 noise_level=self.meta_params['sim_config']['data_noise'],
                                                 seed=self.meta_params['sim_config']['data_seed'])[:]

        val_rates, val_classes = BarsDataset(3, samples_per_class=self.meta_params['sim_config']['n_valid'],
                                             noise_level=self.meta_params['sim_config']['data_noise'],
                                             seed=int(self.meta_params['sim_config']['data_seed'] * 5))[:]

        test_rates, test_classes = BarsDataset(3, samples_per_class=self.meta_params['sim_config']['n_test'],
                                             noise_level=self.meta_params['sim_config']['data_noise'],
                                             seed=int(self.meta_params['sim_config']['data_seed'] * 42))[:]

        indices = np.arange(len(train_classes))
        np.random.shuffle(indices)
        train_rates[:] = np.clip(train_rates[indices, :], 0., 1.)
        train_classes[:] = train_classes[indices]

        indices = np.arange(len(val_classes))
        np.random.shuffle(indices)
        val_rates[:] = np.clip(val_rates[indices, :], 0., 1.)
        val_classes[:] = val_classes[indices]

        indices = np.arange(len(test_classes))
        np.random.shuffle(indices)
        test_rates[:] = np.clip(test_rates[indices, :], 0., 1.)
        test_classes[:] = test_classes[indices]

        classes = np.append(train_classes, val_classes)
        unique = np.unique(classes)

        # teacher signal
        sup_train_rates = np.vstack([(train_classes == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_train_rates[sup_train_rates == 0] = self.meta_params['sim_config']['ulow']
        sup_val_rates = np.vstack([(val_classes == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_val_rates[sup_val_rates == 0] = self.meta_params['sim_config']['ulow']
        sup_test_rates = np.vstack([(test_classes == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_test_rates[sup_test_rates == 0] = self.meta_params['sim_config']['ulow']

        self.meta_params['input_rates'] = {
            Stages.VALIDATION: val_rates.T,
            Stages.TESTING: test_rates.T,
            Stages.TRAINING: train_rates.T,
        }
        self.meta_params['output_rates'] = {
            Stages.VALIDATION: sup_val_rates,
            Stages.TESTING: sup_test_rates,
            Stages.TRAINING: sup_train_rates,
        }

        np.random.seed(self.meta_params['numpy_seed'])
        return

    def load_mimic_dataset(self):
        np.random.seed(self.meta_params['sim_config']['data_seed'])
        train_rates = np.random.uniform(self.meta_params['sim_config']['ulow'],
                self.meta_params['sim_config']['uhigh'],
                size=(self.network_params['dims'][0], self.meta_params['sim_config']['n_train'])).round(4)
        val_rates = np.random.uniform(self.meta_params['sim_config']['ulow'],
                self.meta_params['sim_config']['uhigh'],
                size=(self.network_params['dims'][0], self.meta_params['sim_config']['n_valid'])).round(4)
        test_rates = np.random.uniform(self.meta_params['sim_config']['ulow'],
                self.meta_params['sim_config']['uhigh'],
                size=(self.network_params['dims'][0], self.meta_params['sim_config']['n_test'])).round(4)

        self.meta_params['input_rates'] = {
            Stages.VALIDATION: val_rates,
            Stages.TESTING: test_rates,
            Stages.TRAINING: train_rates,
        }
        combi_params = copy.deepcopy(self.meta_params)
        combi_params.update(copy.deepcopy(self.network_params))
        combi_params['output_rates'] = 'ideal'
        # TODO: hacky way to enforce teacher weight ranges different from student weight ranges
        combi_params['init_weights']['down'] = [-1., 1.]
        combi_params['init_weights']['up'] = [1.99, 2.]
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(combi_params)
        tmp_net = Network(combi_params)
        out_rates = tmp_net.layers[-1].pop.rates
        out_rates = {k: out_rates[k].reshape((self.network_params['dims'][-1], -1)).copy() for k in out_rates}
        self.meta_params['output_rates'] = out_rates
        print('Target weights:', tmp_net.weights)
        del tmp_net
        np.random.seed(self.meta_params['numpy_seed'])

    def load_yin_yang_dataset(self):
        np.random.seed(self.meta_params['sim_config']['data_seed'])
        x_train, y_train = YinYangDataset(size=self.meta_params['sim_config']['n_train'] * 3,
                seed=self.meta_params['sim_config']['data_seed'])[:]
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val, y_val = YinYangDataset(size=self.meta_params['sim_config']['n_valid'] * 3,
                seed=self.meta_params['sim_config']['data_seed'] + 1)[:]
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        x_test, y_test = YinYangDataset(size=self.meta_params['sim_config']['n_test'] * 3,
                seed=self.meta_params['sim_config']['data_seed'] + 2)[:]
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        classes = np.append(y_train, y_val)
        unique = np.unique(classes)
        sup_train_rates = np.vstack([(y_train == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_train_rates[sup_train_rates == 0] = self.meta_params['sim_config']['ulow']
        sup_val_rates = np.vstack([(y_val == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_val_rates[sup_val_rates == 0] = self.meta_params['sim_config']['ulow']
        sup_test_rates = np.vstack([(y_test == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_test_rates[sup_test_rates == 0] = self.meta_params['sim_config']['ulow']

        self.meta_params['input_rates'] = {
            Stages.VALIDATION: x_val.T,
            Stages.TESTING: x_test.T,
            Stages.TRAINING: x_train.T,
        }
        self.meta_params['output_rates'] = {
            Stages.VALIDATION: sup_val_rates,
            Stages.TESTING: sup_test_rates,
            Stages.TRAINING: sup_train_rates,
        }

        np.random.seed(self.meta_params['numpy_seed'])

    def load_mnist_dataset(self):
        import torchvision
        self.data = torchvision.datasets.MNIST('../data/mnist', train=True, download=True)
        y_train = []
        x_train = []
        for i in range(0, 50000, 1):
            y_train.append(self.data.train_labels[i])
            dat_flat = self.data.train_data[i].flatten().cpu().detach().numpy() * 1. / 256.
            x_train.append(dat_flat)
        y_train = np.array(y_train)
        x_train = np.array(x_train)
        y_val = []
        x_val = []
        for i in range(50000, 60000, 1):
            y_val.append(self.data.train_labels[i])
            dat_flat = self.data.train_data[i].flatten().cpu().detach().numpy() * 1. / 256.
            x_val.append(dat_flat)
        y_val = np.array(y_val)
        x_val = np.array(x_val)
        self.data = torchvision.datasets.MNIST('../data/mnist', train=False, download=True)
        y_test = []
        x_test = []
        for i in range(10000):
            y_test.append(self.data.train_labels[i])
            dat_flat = self.data.train_data[i].flatten().cpu().detach().numpy() * 1. / 256.
            x_test.append(dat_flat)
        y_test = np.array(y_test)
        x_test = np.array(x_test)
        classes = np.append(y_train, y_val)
        unique = np.unique(classes)
        sup_train_rates = np.vstack([(y_train == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_train_rates[sup_train_rates == 0] = self.meta_params['sim_config']['ulow']
        sup_val_rates = np.vstack([(y_val == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_val_rates[sup_val_rates == 0] = self.meta_params['sim_config']['ulow']
        sup_test_rates = np.vstack([(y_test == u) * self.meta_params['sim_config']['uhigh'] for u in unique])
        sup_test_rates[sup_test_rates == 0] = self.meta_params['sim_config']['ulow']

        self.meta_params['input_rates'] = {
            Stages.VALIDATION: x_val.T,
            Stages.TESTING: x_test.T,
            Stages.TRAINING: x_train.T,
        }
        self.meta_params['output_rates'] = {
            Stages.VALIDATION: sup_val_rates,
            Stages.TESTING: sup_test_rates,
            Stages.TRAINING: sup_train_rates,
        }
        return


    def eval(self, all_training=False, all_testing=False):
        t1 = time.time()
        an = Analyzer(self.meta_params)
        an.plot_error_signal()
        an.plot_accuracy()
        an.plot_next_ff_vs_p2i()
        an.plot_angle_ff_fb(self.network_params['dims'])
        an.plot_next_fb_vs_i2p()
        an.plot_next_vs_inter_rate()
        #an.plot_signals(Stages.VALIDATION, ['rate'])
        #an.plot_signals(Stages.VALIDATION, ['rate', 'rate_highpass', 'v_apical', 'V', 'v_brev', 'v_basal', 'v_target', 'v_dendrite'])
        if all_training:
            an.plot_signals(Stages.TRAINING, ['g', 'rate', 'rate_highpass', 'v_apical', 'V', 'v_brev', 'v_basal', 'v_target', 'v_dendrite', 'noise'])
        else:
            an.plot_signals(Stages.TRAINING, ['g'])
        if all_testing:
            an.plot_signals(Stages.TESTING, ['rate', 'v_apical', 'V', 'v_brev'])
        else:
            an.plot_signals(Stages.TESTING, ['rate'])
        an.plot_out_vs_sup()
        if self.meta_params['dataset_key'] == 'yinyang':
            an.plot_yy_points()
        t2 = time.time()
        print('Plotting took {} sec'.format(t2-t1))
        return

    def detail_plot(self, epoch):
        an = Analyzer(self.meta_params)
        an.plot_next_ff_vs_p2i_epoch(epoch)
        an.plot_next_vs_inter_rate_epoch(epoch)
        an.plot_signals_epoch(Stages.VALIDATION, epoch, ['rate', 'v_apical', 'V', 'v_brev', 'v_basal', 'v_target', 'v_dendrite'], check_v_dend=False)
        if epoch > 0:
            an.plot_signals_epoch(Stages.TRAINING, epoch, ['g', 'rate', 'v_apical', 'V', 'v_brev', 'v_basal', 'v_target', 'v_dendrite', 'rate_dendrite'])


if __name__ == '__main__':
    import sys
    mode = sys.argv[1]
    if mode == 'train':
        assert len(sys.argv) == 4
        config_path = sys.argv[2]
        basepath = sys.argv[3]
        exp = Experiment(basepath, config_path=config_path)
        exp.run()
    elif mode == 'eval':
        assert len(sys.argv) >= 3
        results_path = sys.argv[2]
        exp = Experiment(results_path, loading_existing=True)
        if len(sys.argv) > 3:
            all_training = bool(int(sys.argv[3]))
            print('Plotting all training traces:', all_training)
        else:
            all_training = False
        if len(sys.argv) > 4:
            all_testing = bool(int(sys.argv[4]))
            print('Plotting all testing traces:', all_testing)
        else:
            all_testing = False

        exp.eval(all_training=all_training, all_testing=all_testing)
    elif mode == 'detail_plot':
        assert len(sys.argv) == 4
        results_path = sys.argv[2]
        epoch = int(sys.argv[3])
        exp = Experiment(results_path, loading_existing=True)
        exp.detail_plot(epoch)
    else:
        raise NotImplementedError


