import numpy as np
import h5py
from os import path
from microcircuits.genn_models.common import Stages


class Recorder:
    def __init__(self, genn_model, layers):
        self.genn_model = genn_model
        self.layers = layers
        self.views = {}
        self.cache = {}
        self.fname = None
        self.stages = [Stages.VALIDATION,
                       Stages.TESTING,
                       Stages.TRAINING]
        self.weights_idx = {k: 0 for k in self.stages}
        self.rates_idx = {k: 0 for k in self.stages}
        self.n_rate_recs_cache = {k: 0 for k in self.stages}
        self.n_weight_recs_cache = 0

    def init(self, config):
        layers = self.layers
        views = {}
        recordings = {}
        h5_fname = config['filename']
        self.fname = h5_fname

        _cfg = config['recordings']
        inds = config['update_indices']

        # these are now split, an epoch is either train or validate
        record_rate_index = inds['rates']
        write_out_index = inds['write']
        record_weight_index = inds['weights']

        total_epoch_steps = inds['train'] + inds['valid']

        n_epochs_per_write = float(write_out_index) / float(total_epoch_steps)

        # ram cache size
        n_epoch_steps = inds['train']
        n_weight_recs = int( np.ceil((n_epochs_per_write * n_epoch_steps) / float(record_weight_index)) )
        n_rate_recs_training = int( np.ceil((n_epochs_per_write * n_epoch_steps) / float(record_rate_index)) )

        n_epoch_steps = inds['valid']
        # ram cache size
        n_rate_recs_validation = int( np.ceil((n_epochs_per_write * n_epoch_steps) / float(record_rate_index)) )
        # n_rate_recs_validation =
        n_epoch_steps = inds['test']
        # ram cache size
        n_rate_recs_test = int( np.ceil((n_epochs_per_write * n_epoch_steps) / float(record_rate_index)) )

        self.n_weight_recs_cache = n_weight_recs
        self.n_rate_recs_cache[Stages.VALIDATION] = n_rate_recs_validation
        self.n_rate_recs_cache[Stages.TRAINING] = n_rate_recs_training

        with h5py.File(h5_fname, 'w') as h5_file:
            for stage in self.stages:
                rs = {}
                vs = {}
                stg_grp = h5_file.create_group(str(stage))
                for lyr_idx in _cfg:
                    rp = {}
                    vp = {}
                    lyr_grp = stg_grp.create_group(str(lyr_idx))
                    lyr = layers[lyr_idx]
                    neurons = lyr.pops
                    synapses = lyr.synapses
                    for pop in _cfg[lyr_idx]:
                        if not(pop in neurons or pop in synapses):
                            print("In Recorder setup: "
                                  "{} not found in layer {}".format(pop, lyr))
                            continue

                        pop_grp = lyr_grp.create_group(pop)

                        if pop in neurons:
                            if stage == Stages.VALIDATION:
                                n_rate_recs = n_rate_recs_validation
                            elif stage == Stages.TESTING:
                                n_rate_recs = n_rate_recs_test
                            else:
                                n_rate_recs = n_rate_recs_training
#                            n_rate_recs = (n_rate_recs_validation
#                                           if stage == Stages.VALIDATION
#                                           else n_rate_recs_training)
                            vp[pop] = {r: neurons[pop].obj.vars[r].view
                                       for r in _cfg[lyr_idx][pop]}
                            rp[pop] = {r: np.zeros((n_rate_recs, vp[pop][r].size))
                                       for r in _cfg[lyr_idx][pop]}
                        elif pop in synapses:
                            vp[pop] = {r: synapses[pop].obj.vars[r].view
                                       for r in _cfg[lyr_idx][pop]}

                            rp[pop] = {r: np.zeros((n_weight_recs, vp[pop][r].size))
                                       for r in _cfg[lyr_idx][pop]}

                        for r in _cfg[lyr_idx][pop]:
                            w = vp[pop][r].shape[0]
                            # print(list(kgrp[r]))
                            pop_grp.create_dataset(r, dtype="float32",
                                                   shape=(1, w),
                                                   maxshape=(None, w),
                                                   chunks=(1, w))

                    vs[lyr_idx] = vp
                    rs[lyr_idx] = rp

                views[stage] = vs
                recordings[stage] = rs

            def get_all(name):
                print(name)

            h5_file.visit(get_all)

        self.views = views
        self.cache = recordings
        # return recordings, views, h5_fname

    def update(self, stage, update_rates, update_weights, write_out=False):
        if write_out:
            print("\n\nWriting to disk...\n")
            self.write_to_file()
            self.clear_recordings()

        if update_rates or update_weights:
            model = self.genn_model
            layers = self.layers
            cache = self.cache
            views = self.views

            for lyr_idx in cache[stage]:
                synapses = layers[lyr_idx].synapses
                neurons = layers[lyr_idx].pops
                for pop in cache[stage][lyr_idx]:
                    if pop in neurons:
                        if not update_rates:
                            continue
                        k = neurons[pop].label
                        model.pull_state_from_device(k)

                    for r in cache[stage][lyr_idx][pop]:
                        if pop in synapses:
                            if not update_weights:
                                continue
                            synapses[pop].obj.pull_var_from_device(r)

                        # rows, cols = cache[lyr_idx][pop][r].shape
                        row = (self.rates_idx[stage]
                               if pop in neurons else
                               self.weights_idx[stage])
                        self.cache[stage][lyr_idx][pop][r][row, :] = \
                                              views[stage][lyr_idx][pop][r]
                        # print(self.cache[lyr_idx][pop][r])

            if update_rates:
                self.rates_idx[stage] += 1

            if update_weights:
                self.weights_idx[stage] += 1

    def write_to_file(self):
        h5_filename = self.fname
        cache = self.cache
        layers = self.layers

        with h5py.File(h5_filename, 'a') as h5_file:
            for stage in cache:
                for lyr_idx in cache[stage]:
                    synapses = layers[lyr_idx].synapses
                    neurons = layers[lyr_idx].pops

                    for pop in cache[stage][lyr_idx]:
                        for r in cache[stage][lyr_idx][pop]:
                            dst = h5_file[path.join(stage, str(lyr_idx), pop, r)]
                            # cache_rows, cache_cols = cache[lyr_idx][pop][r].shape
                            rec_rows = (self.rates_idx[stage]
                                        if pop in neurons else
                                        self.weights_idx[stage])
                            h5_rows, h5_cols = dst.shape
                            if h5_rows == 1:
                                h5_rows = 0
                            dst.resize((h5_rows + rec_rows, h5_cols))

                            dst[h5_rows:, :] = cache[stage][lyr_idx][pop][r][:rec_rows]

            h5_file.flush()

    def clear_recordings(self):
        for stage in self.cache:
            for lyr_idx in self.cache[stage]:
                for pop in self.cache[stage][lyr_idx]:
                    for r in self.cache[stage][lyr_idx][pop]:
                        self.cache[stage][lyr_idx][pop][r][:] = 0

            self.rates_idx[stage] = 0
            self.weights_idx[stage] = 0
                # print(recordings[p][k])
