"""
Weight (synapse) model for (lateral) connections coming from Input or Pyramidals
from a lower layer to a Pyramidal or Output neuron in a higher layer.
"""
import numpy as np
from pygenn.genn_model import (
    create_custom_weight_update_class as custom_w_update,
    create_dpf_class as dpf,
    init_var
)
from microcircuits.genn_models.defaults import run_mode
from microcircuits.genn_models.common import RunModes

__all__ = ["up_synapse_model",]

__vars = [
    ("g", "scalar"),
    ("delta", "scalar"),
]
__params = [
    "tau", # for integration step
    "eta", # learning rate
]

__derived_params = [
    ("integration_step_size", dpf(lambda pars, dt: dt / pars[0])()),
]

__extra_global = [
    ("learning_on", "unsigned char") # to turn on and off learning
]


__code = """
    if ($(learning_on) > 0) {
        scalar ddelta = $(integration_step_size) * (
                -$(delta) +
                DT * $(eta) * ( $(rate_post) - $(rate_basal_post) ) * $(rate_last_pre)
                //( $(rate_post) - $(rate_basal_post) ) * $(rate_last_pre)
        );
        $(g) += $(delta);
        //$(g) += DT * $(eta) * $(delta);

        $(delta) += ddelta;
    }

    // compute input current after weight update
    // printf("\\trate_pre = %f\\t", $(rate_pre));
    $(addToInSyn, $(g)*$(rate_pre));
"""

if run_mode == RunModes.CONTINUOUS:
    sim_code = None
    syndyn_code = __code
else:
    sim_code = __code
    syndyn_code = None

up_synapse_model = custom_w_update(
    "Pyramidal_Up_Synapse",
    param_names=__params,
    var_name_types=__vars,
    derived_params=__derived_params,
    sim_code=sim_code,
    synapse_dynamics_code=syndyn_code,
    extra_global_params=__extra_global,
)
