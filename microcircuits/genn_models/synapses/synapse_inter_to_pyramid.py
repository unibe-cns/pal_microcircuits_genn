"""
Weight (synapse) model for (lateral) connections coming from Inter-neurons to
Pyramidals in the same layer.
"""

import numpy as np
from pygenn.genn_model import (
    create_custom_weight_update_class as custom_w_update,
    create_dpf_class as dpf,
    init_var
)
__all__ = ["i2p_synapse_model"]

from microcircuits.genn_models.defaults import run_mode
from microcircuits.genn_models.common import RunModes

__vars = [
    ("g", "scalar"),
    ("delta", "scalar"),
]
__params = [
    "tau", # for integration time
    "eta", # learning rate
]

__derived_params = [
    ("integration_step_size", dpf(lambda pars, dt: dt / pars[0])()),
]

__extra_global = [
    ("learning_on", "unsigned char") # to turn on and off learning
]


# change decay to integration_time_step
__code = """
    if ($(learning_on) > 0) {
        scalar ddelta = $(integration_step_size) * (
            - $(delta) +
            DT * $(eta) * ((-$(v_apical_post)) * $(rate_last_pre))
            //((-$(v_apical_post)) * $(rate_last_pre))
        );

        //printf("eta %.6f\\t delta %.6f\\t ddelta %.6f\\n",
        //    $(eta), $(delta), ddelta);
        $(g) += $(delta);
        //$(g) += DT * $(eta) * $(delta);

        $(delta) += ddelta;
    }

    // compute input current after weight update
    $(addToInSyn, $(g) * $(rate_pre));
"""

if run_mode == RunModes.CONTINUOUS:
    sim_code = None
    syndyn_code = __code
else:
    sim_code = __code
    syndyn_code = None

i2p_synapse_model = custom_w_update(
    "interToPyramid",
    param_names=__params,
    var_name_types=__vars,
    derived_params=__derived_params,
    sim_code=sim_code,
    synapse_dynamics_code=syndyn_code,
    extra_global_params=__extra_global,
)
