"""
Weight (synapse) model for connections coming from Supervisor to
Output neurons, they communicate a 'teaching signal' which can be turned off
via the learning_on global parameter
"""

import numpy as np
from pygenn.genn_model import (
    create_custom_weight_update_class as custom_w_update,
    create_dpf_class as dpf,
    init_var
)

from microcircuits.genn_models.defaults import run_mode
from microcircuits.genn_models.common import RunModes

__extra_global = [
    ("learning_on", "unsigned char")
]

__code = """
    if($(learning_on) > 0){
        $(addToInSyn, $(g) * $(rate_pre));
    }
"""

if run_mode == RunModes.CONTINUOUS:
    sim_code = None
    syndyn_code = __code
else:
    sim_code = __code
    syndyn_code = None

# because the teacher/supervisor will be a Rate Player neuron model,
# we have to use 'rate' as the
output_down_synapse_model = custom_w_update(
    "Output_Down_Synapse",
    param_names=[],
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code=syndyn_code,
    sim_code=sim_code,
    extra_global_params=__extra_global,
)
