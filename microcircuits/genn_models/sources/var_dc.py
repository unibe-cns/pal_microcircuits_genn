import numpy as np
from pygenn.genn_model import (
    create_custom_current_source_class as custom_source,
    create_dpf_class as dpf,
)
from microcircuits.genn_models.defaults import (
    default_input_target
)

__all__ = ["variable_dc"]


variable_dc = custom_source(
    "variable_dc",
    var_name_types=[
        ("last_t_change", "scalar"),
        ("current_val", "scalar")
    ],
    param_names=[
        "vmin",
        "vmax",
        "period"
    ],
    derived_params=[
        ("magnitude", dpf(lambda pars, dt: pars[1] - pars[0])())
    ],
    injection_code="""
        if($(t) == 0.0f || ($(t) - $(last_t_change)) >= $(period)){{
            $(last_t_change) = $(t);
            $(current_val) = $(vmin) + $(gennrand_uniform) * $(magnitude);
        }}
        {} = $(current_val);
    """.format(default_input_target)
)

