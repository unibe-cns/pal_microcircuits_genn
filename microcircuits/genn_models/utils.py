from pygenn.genn_model import init_var


def list_to_indices(param_list):
    return {i: v for i, v in enumerate(param_list)}


def l2i(param_list):
    return list_to_indices(param_list)


def get_var_init_uniform_sym(a):
    return init_var("Uniform", {'min': -a, 'max': a})

