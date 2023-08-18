"""
Output neuron model with 3 compartments:
    Basal (proximal?).- receives feed-forward data stream through the up synapses
    Apical/Target (distal?).- receives supervision data stream through the
            target synapses
    Soma.- leaky, receives inputs from both feed- forward and back paths through the
           activity on the other compartments.
"""
import numpy as np
from frozendict import frozendict as fzd
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
    "output_neuron_model",
    "out_up_postsyn_model",
    "out_target_postsyn_model",
    "output_act_basal_params",
    "output_act_soma_params",
    "create_output_with_activation",
]

__neuron_model_name = "Output"

__neuron_param_names = [
    # conductances within neurons model
    "g_leak",
    "g_basal",
    "g_target",
    "threshold",
    "tau_highpass",
    "le_active",
]

__neuron_ids = l2i(__neuron_param_names)

__neuron_var_name_types = [
    ("V", "scalar"),  # soma
    ("I", "scalar"),  # soma
    ("v_target", "scalar"),
    ("v_basal", "scalar"),
    ("v_steady", "scalar"),
    ("v_dot", "scalar"),
    ("v_brev", "scalar"),
    ("dv", "scalar"),
    ("RefracTime", "scalar"),
    ("rate", "scalar"),
    ("rate_last", "scalar"),
    ("rate_basal", "scalar"),
    ("rate_highpass", "scalar"),
]
__global_params = [
    ("learning_on", "unsigned char")
]

output_act_soma_params = ["$(v_brev)", "$(rate)", "$(threshold)"]
output_act_basal_params = ["v_rate_b", "$(rate_basal)", "$(threshold)"]
act_soma = default_activation(*output_act_soma_params)
act_basal = default_activation(*output_act_basal_params)

__neuron_sim_code = """
    $(v_basal) = $(Isyn_up); // feedforward
    $(v_target) = $(Isyn_target); // desired class/activity
    scalar dr_hp = $(rate) - $(rate_last) -  DT / $(tau_highpass) * $(rate_highpass);
    $(rate_highpass) += dr_hp;
    $(rate_last) = $(rate);

    if ( $(learning_on) > 0) {{
        $(v_steady) = ($(g_basal) * $(v_basal) + $(g_target) * $(v_target)) / ($(g_basal) + $(g_leak) + $(g_target));
    }}
    else {{
        $(v_steady) = ($(g_basal) * $(v_basal)) / ($(g_basal) + $(g_leak));
    }}
    scalar v_rate_b = $(g_scale) * $(v_basal);

    $(v_dot) = ($(g_basal) + $(g_target) + $(g_leak)) * ($(v_steady) - $(V));
    $(dv) = DT * $(v_dot);
    if ( $(le_active) > 0) {{
        $(v_brev) = $(V) + $(tau_soma) * $(v_dot);
    }}

    $(V) += $(dv);

    if ( $(le_active) == 0) {{
        $(v_brev) = $(V);
    }}

    {}

    {}
"""

__neuron_reset_code = ""

__neuron_threshold_code = default_threshold(0, 0)

__neuron_derived_parameters = [
    ("g_scale", dpf(lambda pars, dt: pars[1] / (pars[0]+pars[1]))()),
    ("tau_soma", dpf(lambda pars, dt: 1./np.sum(pars[:3]))()),
]

__neuron_additional_input_vars = [
    ("Isyn_up", "scalar", 0.0), # feedforward
    ("Isyn_target", "scalar", 0.0), # supervision
]


def create_output_with_activation(
        activations=fzd({'soma': act_soma, 'basal': act_basal}),
        name=__neuron_model_name,
        param_names=__neuron_param_names,
        var_name_types=__neuron_var_name_types,
        sim_code=__neuron_sim_code,
        reset_code=__neuron_reset_code,
        threshold_code=__neuron_threshold_code,
        derived_params=__neuron_derived_parameters,
        additional_inputs=__neuron_additional_input_vars,
        global_params=__global_params):
    sim_plus_act = sim_code.format(activations['soma'],
                                   activations['basal'])
    return custom_neuron(name,
                         param_names=param_names,
                         var_name_types=var_name_types,
                         sim_code=sim_plus_act,
                         reset_code=reset_code,
                         threshold_condition_code=threshold_code,
                         derived_params=derived_params,
                         additional_input_vars=additional_inputs,
                         extra_global_params=global_params)


output_neuron_model = create_output_with_activation()

"""
-----------------------------------------------------------
post-synaptic model
seems to be just a delta post-synapse shape (after the activation function)
-----------------------------------------------------------
"""

# to basal compartment
out_up_postsyn_model = custom_postsynaptic(
    "OutUp_PostSyn",
    apply_input_code="""
        // printf("inSyn = %f\\n", $(inSyn));
        $(Isyn_up) += $(inSyn);
        $(inSyn) = 0;
    """
)

# to apical/target compartment
out_target_postsyn_model = custom_postsynaptic(
    "OutTarget_PostSyn",
    apply_input_code="""
        $(Isyn_target) += $(inSyn);
        $(inSyn) = 0;
    """
)
