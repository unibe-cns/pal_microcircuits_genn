import numpy as np
from microcircuits.genn_models.utils import l2i
from pygenn.genn_model import (
    create_custom_neuron_class as custom_neuron,
    create_dpf_class as dpf,
)


__all__ = [
    "rate_player_model"
]

__model_name = "ratePlayer"

__param_names = [
    "period", # how long each sample runs ms
    "tau", # low-pass filter time constant
    "mute_t", # ms
    "noise_amp",
    "base_rate",
]

__var_name_types = [
    ("rate", "scalar"),
    ("rate0", "scalar"),
    ("rate_last", "scalar"),
    ("last_t", "scalar"),
    ("startStim", "unsigned int"),
    ("endStim", "unsigned int"),
    ("index", "unsigned int"),
    ("step_counter", "unsigned int"),
]

__derived_params = [
    ("decay_step", dpf(lambda pars, dt: dt / pars[1])()),
    ("iperiod", # index version of period; for comparisons
     dpf(lambda pars, dt: int(float(pars[0])/dt))()),
    ("istart", # index version of start time; for comparisons
     dpf(lambda pars, dt: int(pars[2]/dt))()),
    ("imute", # index version of mute time; for comparisons
     dpf(lambda pars, dt: int(pars[3]/dt))()),
]
__extra_global_params = [
    ("rates_list", "scalar*"),
    ("rate_shuffler", "int*")
]

__sim_code = """
    $(rate_last) = $(rate);
    unsigned int sdt = (unsigned int)( ($(t) - $(last_t)) / DT );
    unsigned int iiperiod = (unsigned int)$(iperiod);
    unsigned int iimute = (unsigned int)$(imute);

    // when the period for the current rate is done
    if ( $(step_counter) >= iiperiod ) {
        $(index)++; // advance (mean) rate value pointer
        $(rate0) = $(rates_list)[$(startStim) + $(rate_shuffler)[$(index)]]; // grab value and store
        $(last_t) = $(t); // save when the last rate switch was made
        // check if a new epoch is starting, restart inputs
        if( ($(startStim) + $(index)) == $(endStim) ){
            $(index) = 0;
       }
       $(step_counter) = 0;
    }

    scalar r = $(rate0);
    // if we want to keep the player silent for some period before starting
    if ( $(step_counter) < iimute ) {
        r = $(base_rate);
    }

    scalar drate = $(decay_step) * (r - $(rate_last));
    $(rate) = $(rate_last) + drate;
    $(step_counter) = $(step_counter) + 1;
"""

rate_player_model = custom_neuron(
    __model_name,
    param_names=__param_names,
    derived_params=__derived_params,
    var_name_types=__var_name_types,
    sim_code=__sim_code,
    extra_global_params=__extra_global_params
)
