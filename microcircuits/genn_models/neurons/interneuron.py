"""
Inter neuron model with 3 compartments:
    Basal (soma in this code!). Receives input from the pyramidal neurons in the
        same layer; this could proabably be still seen as a 'feed-forward' path.
    Dendrite (distal?). Receives feed-back (prediction?) data stream from the
        next higher layer  through the down synapses and from interneurons
        through the inter synapses.
    Soma. Leaky, receives inputs from both feed- forward and back paths through
        the activity on the other compartments.

The populations for this neuron type are (typically) of same size of the next
higher layer and the connectivity (to dendrites) is one-to-one. This is also a
rather odd connection in the sense that the feed-back signal is just the voltage
from a pair neuron in the next layer.
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
    "inter_neuron_model",
    "inter_pyramidal_postsyn_model",
    "inter_down_postsyn_model",
    "interneuron_act_soma_args",
    "interneuron_act_dendrite_args",
    "create_interneuron_with_activation"
]

__neuron_model_name = "Interneuron"
__neuron_param_names = [
    # conductances within neurons model
    "g_leak",
    "g_dendrite",
    "g_soma",
    "threshold",
    "noise_amplitude",
    "le_active",
]

__neuron_ids = l2i(__neuron_param_names)

__neuron_param_names_types = [
    ("g_leak", "scalar"),
    ("g_dendrite", "scalar"),
    ("g_soma", "scalar"),
    ("threshold", "scalar"),
    ("noise_amplitude", "scalar"),
    ("le_active", "scalar"),        # if le is active, copy V into v_brev -> then orig MC
]

__neuron_var_name_types = [
    ("V", "scalar"),  # soma
    ("I", "scalar"),  # soma
    ("v_down", "scalar"),
    ("v_dendrite", "scalar"),
    ("v_steady", "scalar"),
    ("v_dot", "scalar"),
    ("v_brev", "scalar"),
    ("RefracTime", "scalar"),
    ("dv", "scalar"),
    ("rate", "scalar"),
    ("rate_last", "scalar"),
    ("rate_dendrite", "scalar"),
    ("rate_dendrite_last", "scalar"),

]

interneuron_act_soma_args = ["$(v_brev)", "$(rate)", "$(threshold)"]
interneuron_act_dendrite_args = ["v_rate_d", "$(rate_dendrite)", "$(threshold)"]
act_soma = default_activation(*interneuron_act_soma_args)
act_dend = default_activation(*interneuron_act_dendrite_args)

__neuron_sim_code = """
    $(rate_last) = $(rate);
    $(rate_dendrite_last) = $(rate_dendrite);
    $(v_dendrite) = $(Isyn_pyramidal);
    $(v_down) = $(Isyn_down);
    $(v_steady) = ($(g_dendrite) * $(v_dendrite) + $(g_soma) * $(v_down)) * $(tau_soma);
    scalar v_rate_d = $(g_scale) * $(v_dendrite);

    $(v_dot) = ($(g_dendrite) + $(g_soma) + $(g_leak)) * ($(v_steady) - $(V));
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
"""#.format(act_soma, act_dend)

__neuron_reset_code = ""

__neuron_threshold_code = default_threshold(0, 0)

__neuron_derived_parameters = [
    ("g_scale", dpf(lambda pars, dt: pars[1] / (pars[0] + pars[1]))()),
    ("tau_soma", dpf(lambda pars, dt: 1./np.sum(pars[:3]))()),
]

__neuron_additional_input_vars = [
    ("Isyn_down", "scalar", 0.0), # feedback
    ("Isyn_pyramidal", "scalar", 0.0), # lateral, comes from (all) pyramidals in same layer
]

__neuron_extra_global_parameters = [
    ("spikeTimes", "scalar*"),
]


def create_interneuron_with_activation(
        activations=fzd({'soma': act_soma, 'dendrite': act_dend}),
        name=__neuron_model_name,
        param_names=__neuron_param_names,
        var_name_types=__neuron_var_name_types,
        sim_code=__neuron_sim_code,
        reset_code=__neuron_reset_code,
        threshold_code=__neuron_threshold_code,
        derived_params=__neuron_derived_parameters,
        additional_inputs=__neuron_additional_input_vars):
    sim_plus_act = sim_code.format(activations['soma'],
                                   activations['dendrite'])
    return custom_neuron(name,
                         param_names=param_names,
                         var_name_types=var_name_types,
                         sim_code=sim_plus_act,
                         reset_code=reset_code,
                         threshold_condition_code=threshold_code,
                         derived_params=derived_params,
                         additional_input_vars=additional_inputs)


inter_neuron_model = create_interneuron_with_activation()

"""
-----------------------------------------------------------
post-synaptic model
seems to be just a delta post-synapse shape (after the activation function)
-----------------------------------------------------------
"""

inter_down_postsyn_model = custom_postsynaptic(
    "InterDown",
    apply_input_code="""
        $(Isyn_down) += $(inSyn);
        $(inSyn) = 0;
    """
)

inter_pyramidal_postsyn_model = custom_postsynaptic(
    "InterPyr",
    apply_input_code="""
         $(Isyn_pyramidal) += $(inSyn);
         $(inSyn) = 0;
    """
)

