"""
Pyramidal neuron model with 3 compartments:
    Basal (proximal?).- receives feed-forward data stream through the up synapses
    Apical (distal?).- receives feed-back (prediction?) data stream through the
            down synapses and from interneurons through the inter synapses
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
    "pyramidal_neuron_model",
    "pyr_up_postsyn_model",
    "pyr_down_postsyn_model",
    "pyr_inter_postsyn_model",
    "create_pyramidal_with_activation",
    "pyramidal_act_basal_params",
    "pyramidal_act_soma_params",
]

__neuron_model_name = "Pyramidal"

__neuron_param_names = [
    # conductances within neurons model
    "g_leak",
    "g_basal",
    "g_apical",
    "threshold",
    "sigma_noise",
    "tau_noise",
    "tau_highpass",
    "le_active",
]

__neuron_ids = l2i(__neuron_param_names)

__neuron_param_names_types = [
    ("g_leak", "scalar"),
    ("g_basal", "scalar"),
    ("g_apical", "scalar"),
    ("threshold", "scalar"),
    ("sigma_noise", "scalar"),
    ("tau_noise", "scalar"),
    ("tau_highpass", "scalar"),
    ("le_active", "scalar"),        # if le is not active, copy V into v_brev -> then orig MC
]

__neuron_var_name_types = [
    ("V", "scalar"),  # soma
    ("I", "scalar"),  # soma
    ("v_apical", "scalar"),
    ("v_basal", "scalar"),
    ("v_steady", "scalar"),
    ("noise", "scalar"),
    ("dv", "scalar"),
    ("v_dot", "scalar"),
    ("v_brev", "scalar"),
    ("RefracTime", "scalar"),
    ("rate", "scalar"),
    ("rate_last", "scalar"),
    ("rate_basal", "scalar"),
    ("rate_highpass", "scalar"),
]

pyramidal_act_soma_params = ["$(v_brev)", "$(rate)", "$(threshold)"]
pyramidal_act_basal_params = ["v_rate_b", "$(rate_basal)", "$(threshold)"]
act_soma = default_activation(*pyramidal_act_soma_params)
act_basal = default_activation(*pyramidal_act_basal_params)

__neuron_sim_code = """
    $(v_basal) = $(Isyn_up);
    $(v_apical) = $(Isyn_down) + $(Isyn_inter);
    scalar dnoise = 1. / $(tau_noise) * ($(scale_noise) * $(gennrand_normal) - DT * $(noise));
    $(noise) += dnoise;
    scalar dr_hp = $(rate) - $(rate_last) -  DT / $(tau_highpass) * $(rate_highpass);
    $(rate_highpass) += dr_hp;
    $(rate_last) = $(rate);
    $(v_steady) = ($(g_basal) * $(v_basal) + $(g_apical) * ($(v_apical) + $(noise))) * $(tau_soma);

    scalar v_rate_b = $(g_scale) * $(v_basal);

    $(v_dot) = ($(g_basal) + $(g_apical) + $(g_leak)) * ($(v_steady) - $(V));
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
    # ("ExpTC", dpf(lambda pars, dt: np.exp(-dt / pars[1]))()),
    ("g_total", dpf(lambda pars, dt: np.sum(pars[:3]))()),
    ("g_scale", dpf(lambda pars, dt: pars[1]/np.sum(pars[:3]))()),
    ("tau_soma", dpf(lambda pars, dt: 1./np.sum(pars[:3]))()),
    ("scale_noise", dpf(lambda pars, dt: np.sqrt(dt * pars[5]) * pars[4])()),
]

__neuron_additional_input_vars = [
    ("Isyn_up", "scalar", 0.0), # current coming from the feed-forward path
    ("Isyn_down", "scalar", 0.0), # current coming from the feed-back path
    ("Isyn_inter", "scalar", 0.0), # current coming from the interneuron loop
]

__neuron_extra_global_parameters = [
    ("spikeTimes", "scalar*"),
]


def create_pyramidal_with_activation(
        activations=fzd({'soma': act_soma, 'basal': act_basal}),
        name=__neuron_model_name,
        param_names=__neuron_param_names,
        var_name_types=__neuron_var_name_types,
        sim_code=__neuron_sim_code,
        reset_code=__neuron_reset_code,
        threshold_code=__neuron_threshold_code,
        derived_params=__neuron_derived_parameters,
        additional_inputs=__neuron_additional_input_vars):
    sim_plus_act = sim_code.format(activations['soma'],
                                   activations['basal'])
    return custom_neuron(name,
                         param_names=param_names,
                         var_name_types=var_name_types,
                         sim_code=sim_plus_act,
                         reset_code=reset_code,
                         threshold_condition_code=threshold_code,
                         derived_params=derived_params,
                         additional_input_vars=additional_inputs)


pyramidal_neuron_model = create_pyramidal_with_activation()


"""
-----------------------------------------------------------
post-synaptic model
seems to be just a delta post-synapse shape (after the activation function)
-----------------------------------------------------------
"""

# to basal compartment
pyr_up_postsyn_model = custom_postsynaptic(
    "PyrUp_PostSyn",
    apply_input_code="""
        $(Isyn_up) += $(inSyn);
        $(inSyn) = 0;
    """
)

# To apical compartment
pyr_down_postsyn_model = custom_postsynaptic(
    "PyrDown_PostSyn",
    apply_input_code="""
        $(Isyn_down) += $(inSyn);
        $(inSyn) = 0;
    """
)

# To apical compartment
pyr_inter_postsyn_model = custom_postsynaptic(
    "PyrInter_PostSyn",
    apply_input_code="""
        $(Isyn_inter) += $(inSyn);
        $(inSyn) = 0;
    """
)
