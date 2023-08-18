import numpy as np
from microcircuits.genn_models.utils import l2i
from pygenn.genn_model import (
    create_custom_neuron_class as custom_neuron,
    create_dpf_class as dpf,
    create_custom_postsynaptic_class as custom_postsynaptic,
)
from microcircuits.genn_models.defaults import (
    default_activation, default_threshold
)

__all__ = [
    "variable_rate_source_model",
]

variable_rate_source_model = custom_neuron(
    "variable_rate_source",
    var_name_types=[
        ("last_t_change", "scalar"),
        ("rate", "scalar"),
        ("rate_last", "scalar")
    ],
    param_names=[
        "vmin",
        "vmax",
        "period"
    ],
    derived_params=[
        ("magnitude", dpf(lambda pars, dt: pars[1] - pars[0])())
    ],
    sim_code="""
        $(rate_last) = $(rate);
        if($(t) == 0.0f || ($(t) - $(last_t_change)) >= $(period)){
            $(last_t_change) = $(t);
            $(rate) = $(vmin) + $(gennrand_uniform) * $(magnitude);
        }
    """
)

