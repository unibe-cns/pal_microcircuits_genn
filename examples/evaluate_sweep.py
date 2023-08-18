#import h5py
import sys
import yaml
import shutil
import numpy as np
import matplotlib.pyplot as plt
from experiment_runner import Experiment
from microcircuits.genn_models import Analyzer

def load_file_list(file_list, save_dir):
    shutil.copyfile(file_list, save_dir + '/' + file_list)
    with open(file_list, 'r') as f:
        sweep_dirs = f.readlines()
    sweep_dirs = [sweep[:-1] for sweep in sweep_dirs]
    print(sweep_dirs)
    return sweep_dirs


def load_config(dirname):
    path = dirname + '/' + 'config.yaml'
    with open(path) as f:
        data = yaml.load_all(f, Loader=yaml.Loader)
        all_configs = next(iter(data))
    meta_params = all_configs[0]
    network_params = all_configs[1]
    return meta_params, network_params


def load_analyzers(sweep_dirs):
    exp_list = [Experiment(sweep, loading_existing=True) for sweep in sweep_dirs]
    analyzer_list = [Analyzer(exp.meta_params) for exp in exp_list]
    return exp_list, analyzer_list


def collect_mse(save_dir, analyzer_list, sweep_dirs):
    mse_collection = []
    for a in analyzer_list:
        print('loading mse from {}'.format(sweep_dirs[i]), flush=True)
        mse = a.get_error_signal()
        mse_collection.append(mse)
    np.save(save_dir + '/mse_collection.npy', mse_collection)
    return np.array(mse_collection)


def calc_test_acc(analyzer):
    rec_idx = analyzer.config['sim_config']['idx_rec']
    n_per_sample = analyzer.config['sim_config']['recs_per_sample']
    n_per_epoch = analyzer.config['sim_config']['idx_test'] // rec_idx
    n_tests = float(n_per_epoch // n_per_sample)
    out, sup = analyzer.get_out_and_sup_signals(which_stage='testing')
    accuracy = []
    assert n_per_epoch == out.shape[0]
    correct = 0
    for si in range(0, n_per_epoch, n_per_sample):
        ei = si + n_per_sample
        m_out = np.mean(out[si+1:ei], axis=0)
        m_sup = np.max(sup[si+1:ei], axis=0)
        out_win = np.argmax(m_out)
        sup_win = np.argmax(m_sup)
        if out_win == sup_win:
            correct += 1
    accuracy = 100. * correct/n_tests
    return accuracy


def collect_accs(save_dir, analyzer_list, sweep_dirs, recalc):
    acc_collection = []
    if recalc:
        for i, a in enumerate(analyzer_list):
            print('loading accuracy from {}'.format(sweep_dirs[i]), flush=True)
            acc = a.get_accuracy_signal()
            acc_collection.append(acc)
        np.save(save_dir + '/acc_collection.npy', acc_collection)
    else:
        try:
            print('Reloading precalculated accuracy collection', flush=True)
            acc_collection = np.load(save_dir + '/acc_collection.npy')
        except:
            print('Tried reloading precalculated accuracy collection, did not work, calculating it now', flush=True)
            acc_collection = collect_accs(save_dir, analyzer_list, sweep_dirs, True)
    return np.array(acc_collection)


def collect_test_accs(save_dir, analyzer_list, sweep_dirs, recalc):
    test_acc_collection = []
    if recalc:
        for i, a in enumerate(analyzer_list):
            print('Calculating test accuracy from {}'.format(sweep_dirs[i]), flush=True)
            test_acc = calc_test_acc(a)
            test_acc_collection.append(test_acc)
        np.save(save_dir + '/test_acc_collection.npy', test_acc_collection)
    else:
        try:
            print('Reloading precalculated test accuracy collection', flush=True)
            test_acc_collection = np.load(save_dir + '/test_acc_collection.npy')
        except:
            print('Tried reloading precalculated test accuracy collection, did not work, calculating it now', flush=True)
            test_acc_collection = collect_test_accs(save_dir, analyzer_list, sweep_dirs, True)
    return test_acc_collection


def collect_angles(save_dir, analyzer_list, sweep_dirs, recalc):
    # TODO: extend to multiple hidden layers
    layer_idx = '1'
    if recalc:
        angle_collection = []
        for i, a in enumerate(analyzer_list):
            print('Calculating BPP-WPP angle from {}'.format(sweep_dirs[i]), flush=True)
            meta_params, net_params = load_config(sweep_dirs[i])
            angle_dict = a.plot_angle_ff_fb(net_params['dims'], return_angle=True)
            angle_collection.append(angle_dict[layer_idx])
        np.save(save_dir + '/angle_collection_layer_{}.npy'.format(layer_idx), angle_collection)
    else:
        try:
            print('Reloading precalculated angle collection')
            angle_collection = np.load(save_dir + '/angle_collection_layer_{}.npy'.format(layer_idx))
        except:
            print('Tried reloading precalculated angle collection, did not work, calculating it now', flush=True)
            angle_collection = collect_angles(save_dir, analyzer_list, sweep_dirs, True)
    return angle_collection


def plot_accs(save_dir, acc_collection, test_acc_collection):
    plt.figure()
    for i, acc in enumerate(acc_collection):
        plt.plot(acc, alpha=0.7, color='C{}'.format(i))
        #plt.axhline(test_acc_collection[i], alpha=0.7, ls='--', color='C{}'.format(i))
    plt.errorbar(len(acc) + len(acc) // 100 * 5, np.mean(test_acc_collection), yerr=np.std(test_acc_collection),
            label='mean test acc: {0:.1f} +- {1:.1f}'.format(np.mean(test_acc_collection), np.std(test_acc_collection)),
            marker='o', color='gray', capsize=4, elinewidth=2)
    plt.xlabel('epochs')
    plt.ylabel('validation accuracy [%]')
    plt.legend()
    plt.savefig(save_dir + '/acc_collection.png')
    return


def plot_angles(save_dir, angle_collection):
    plt.figure()
    for i, a in enumerate(angle_collection):
        plt.plot(a, alpha=0.7, color='C{}'.format(i))
    #plt.xlabel('epochs')
    plt.ylabel('Angle WPP-BPP [°]')
    plt.legend()
    plt.savefig(save_dir + '/angle_collection.png')
    return



if __name__ == '__main__':
    file_list = sys.argv[1]
    save_dir = sys.argv[2]
    recalc = True if sys.argv[3] == 'recalc' else False
    sweep_dirs = load_file_list(file_list, save_dir)
    exp_list, analyzer_list = load_analyzers(sweep_dirs)
    acc_collection = collect_accs(save_dir, analyzer_list, sweep_dirs, recalc)
    test_acc_collection = collect_test_accs(save_dir, analyzer_list, sweep_dirs, recalc)
    plot_accs(save_dir, acc_collection, test_acc_collection)
    angle_collection = collect_angles(save_dir, analyzer_list, sweep_dirs, recalc)
    plot_angles(save_dir, angle_collection)

