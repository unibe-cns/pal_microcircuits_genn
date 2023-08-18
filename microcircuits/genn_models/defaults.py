from .common import (
    activation_templates, act_numpy, threshold_templates,
    RunModes
)
default_activation_name = "soft_relu"
default_activation = activation_templates[default_activation_name]
default_act_numpy = act_numpy[default_activation_name]
default_threshold = threshold_templates["none"]
default_input_target = "Isyn"
run_mode = RunModes.CONTINUOUS
