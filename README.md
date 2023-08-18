# Dendritic Microcircuits implementation for the GeNN simulator

This repository is part of the code-base for the paper
[Learning efficient backprojections across cortical hierarchies in real time](https://arxiv.org/abs/2212.10249) and its algorithm for learning feedback weights in deep networks "PAL".

Here we demonstrate the capabilities of PAL in a biologically plausible network
setting using the model of the [Dendritic Microcircuits](https://papers.nips.cc/paper/2018/hash/1dc3a89d0d440ba31729b0ba74b93a33-Abstract.html) 
extended by the [Latent Equilibrium-](https://proceedings.neurips.cc/paper_files/paper/2021/hash/94cdbdb84e8e1de8a725fa2ed61498a4-Abstract.html) and
PAL-mechanisms.

## Base model
The basis of this repository was [Gary Garcia's](https://github.com/chanokin) [GeNN](https://github.com/genn-team/genn) based implementation of the [Dendritic cortical microcircuits 
approximate the backpropagation algorithm](https://papers.nips.cc/paper/2018/hash/1dc3a89d0d440ba31729b0ba74b93a33-Abstract.html) paper.
It included the GeNN-specific implementation of neuron and synapse dynamics as
well as the network structure and recording infrastructure. Since this code is
still under active development, the original repository is not publically
available yet.

## Extensions for Latent Equilibrium and PAL
Starting from a snapshot of the base model code, this repository extends the
original dendritic microcircuit model significantly by including the Latent
equilibrium dynamics as well as the infrastructure and dynamics for learning the
feedback weight using PAL. Additionally, infrastrucure for experiment
configuration, running, evaluation, parameter sweeps, etc were added.

### Structure:
```
.
|-- examples                <== use cases
|   |-- experiment_configs  <== config files to reproduce paper results
|   `-- code for running experiments  <== Infrastructure to run and evaluate experiments
`-- microcircuits
    `-- genn_models     <== main library location
        |-- neurons     <== 'low-level' GeNN representations for neurons
        |-- synapses    <== 'low-level' GeNN representations for synapses
        `-- wrappers    <== high-level representations (e.g. layers, network)
```

## Installation

This code uses the GeNN simulator and its Python interface PyGenn. It was
developed and tested with GeNN 4.6.0 and the corresponding version of PyGenn
released with it. You can try to run this code on other GeNN versions, but this
has not been tested so far.

- make new venv with Python 3.6: `virtualenv --python=python3.6 name`
- download GeNN 4.6.0 zip from [GeNN releases](https://github.com/genn-team/genn/releases/tag/4.6.0) and unzip
- download PyGenn wheel (.whl) from [GeNN releases](https://github.com/genn-team/genn/releases/tag/4.6.0)
- `pip install pygenn-4.6.0-cp36-cp36m-linux_x86_64.whl`
- set paths in `virtual/bin/activate`:
 ```
export CUDA_PATH=/usr/local/cuda-11.0
export PATH=$PATH:$CUDA_PATH/bin
export PATH=$PATH:/users/.../genn/genn-4.6.0/bin
export PYTHONPATH=/users/.../pal_microcircuits_genn
 ```
 - navigate to the repository and install requirements: `pip install -r requirements.txt`. If this fails, pip install modules manually.

## The experiment runner setup:

#### How to run an experiment
- Navigate to the `examples/` subdirectory
- Use `python experiment_runner.py train <path_to_config_file>
  <where_to_save_data>`
- This will create a directory in `<where_to_save_data>` with the name specified
  in the config file extended by a time stamp
- In that directory all recorded variables (the config file specifies which
  variables are recorded) are saved as `net_recording.h5`. Note the the more is
  recorded, the slower the simulation gets. Also note that recording e.g.
  weights can easily cause the results directory to contain multiple GB of data.
- Additionally the gitsha of the version of the code that is used to simulate is
  written to the results directory.
- The config file given in the run command is loaded by the simulator, extended
  by the values that are generated at runtime and then also written into the
  results directory (so that you can later look up which parameters produced the
  result).

#### How to evaluate an experiment

- Navigate to the `examples/` subdirectory
- Use `python experiment_runner.py eval <path_to_saved_results> 0/1 0/1`
- The last two arguments determine how much is plotted: the first 0 or 1 toggles
  plotting of the variables recorded during training, the second during the test
  run at the end
- All generated plots are saved alongside the simulation results
- To control in more detail what is plotted mess around with the funtion
  `eval()` in the `Experiment` class in `experiment_runner.py`
- In addition to the general evaluation detailed evaluations of specific epochs
  can be run with `python experiment_runner.py detail_plot <path_to_saved_results> <epoch_nr>`
- These plots are saved in subdirectories of the results directory
- Note that due to the way validation and training runs are interleaved, epoch 0
  contains only a validation run (the very first one before training started)
  and all following epochs contain training results and the validation after
  that training epoch.

#### Config files

- Config files are located in `examples/experiment_configs/` as yaml files
- To create a new one copy an existing yaml file and adapt the content
- To generate a set of config files for a sweep over seeds use the
  `generate_sweep_cfg_files.py`.

#### Datasets

- Which dataset is chosen for a run is determined by the value of `dataset_key`
  in the meta-parameters in the config file.
- The experiment runner reads that parameter and according to its value loads
  the corresponding dataset
- To add support of a new dataset give it name that is used as `dataset_key` and
  define how it should be loaded in the Experiment class in the function
  `load_dataset()`.

#### Recreate paper plots

- All config files required to recreate the figures in the paper are stored in
  the `examples/experiment_configs/` directory.
- To create the yin-yang dataset comparison between FA and PAL run an experiment for all config
  files in the directories `examples/experiment_configs/sweep_yin_yang_fa/` and `examples/experiment_configs/sweep_yin_yang_backw/`
- To create the mnist dataset comparison between FA and PAL run an experiment for all config
  files in the directories `examples/experiment_configs/sweep_mnist_fa/` and `examples/experiment_configs/sweep_mnist_backw/`
- To create the supplementary plot that compares PAL with a PAL run where the
  feedback-weight learning rate is set to zero (to exclude that the advantage
  over FA is due to noise) run an experiment for all config files in the directories `examples/experiment_configs/sweep_yin_yang_bw_eta_zero/`
  and `examples/experiment_configs/sweep_yin_yang_backw/`
- To evaluate the sweeps and create summary plots and data use the
  `examples/evaluate_sweep.py` script. For that assemble all results-directories to be
  evaluated in a list in a .txt file.

