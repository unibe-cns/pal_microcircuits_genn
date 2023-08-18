"""
Weight (synapse) model for connections coming from Output or Pyramidal neurons
in a higher layer to Inter-neurons. These are kept static for our implementation.
"""
import numpy as np
from pygenn.genn_model import (
    create_custom_weight_update_class as custom_w_update,
    create_dpf_class as dpf,
    init_var
)

from microcircuits.genn_models.common import RunModes
from microcircuits.genn_models.defaults import run_mode

__code = "$(addToInSyn, $(g) * $(v_brev_pre));"

if run_mode == RunModes.CONTINUOUS:
    sim_code = None
    syndyn_code = __code
else:
    sim_code = __code
    syndyn_code = None


inter_down_synapse_model = custom_w_update(
    "interNeuronDownSynapse",
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code=syndyn_code,
    sim_code=sim_code,
)

