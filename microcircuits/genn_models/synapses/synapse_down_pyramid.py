"""
Weight (synapse) model for connections coming from Pyramidals in higher layers
to Pyramidals in lower layers. These remain static in our implementation.
"""
import numpy as np
from pygenn.genn_model import (
    create_custom_weight_update_class as custom_w_update,
    create_dpf_class as dpf,
    init_var
)

from microcircuits.genn_models.defaults import run_mode
from microcircuits.genn_models.common import RunModes

__vars = [
    ("g", "scalar"),    # synaptic weight
]

__params = [
    "alpha", # for balancing regularizer
    "eta",   # learning rate
]

__extra_global = [
    ("learning_on", "unsigned char") # to turn on and off learning
]

__code = """
    if ($(learning_on) > 0) {
        scalar dg = DT * $(eta) * ($(noise_post) * $(rate_highpass_pre) - $(alpha) * $(g));
        $(g) += dg;
    }
    $(addToInSyn, $(g) * $(rate_pre));
"""

if run_mode == RunModes.CONTINUOUS:
    sim_code = None
    syndyn_code = __code
else:
    sim_code = __code
    syndyn_code = None

pyramidal_down_synapse_model = custom_w_update(
    "Pyramidal_Down_Synapse",
    param_names=__params,
    var_name_types=__vars,
    sim_code=sim_code,
    synapse_dynamics_code=syndyn_code,
    extra_global_params=__extra_global,
)
