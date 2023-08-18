import numpy as np
from microcircuits.genn_models.utils import l2i
from pygenn.genn_model import (
    create_custom_neuron_class as custom_neuron,
    create_dpf_class as dpf,
    # create_custom_postsynaptic_class as custom_postsynaptic,
)
# from microcircuits.genn_models.defaults import (
#     default_activation, default_threshold
# )

__all__ = [
    "bias_model"
]

__model_name = "bias"

__param_names = [
    "value",
]

__var_name_types = [
    ("rate", "scalar"),
    ("rate_last", "scalar"),
]

__derived_params = [
]

__extra_global_params = [
]

__sim_code = """
    $(rate) = $(value);
    $(rate_last) = $(value);
"""

bias_model = custom_neuron(
    __model_name,
    param_names=__param_names,
    derived_params=__derived_params,
    var_name_types=__var_name_types,
    sim_code=__sim_code,
    extra_global_params=__extra_global_params
)
