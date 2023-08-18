from typing import NamedTuple
import numpy as np


class RUN_MODES(NamedTuple):
    CONTINUOUS:str = "continuous"
    EVENTS:str = "events"


class STAGES(NamedTuple):
    VALIDATION:str = "validation"
    TRAINING:str = "training"
    TESTING:str = "testing"

class OBJECT_TYPES(NamedTuple):
    NEURONS:str = "neurons"
    SYNAPSES:str = "synapses"

def soft_relu(x, thresh=15):
    res = x.copy()
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] = np.log(1 + np.exp(x[ind]))
    return res

# faster than the stable version
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_stable(x, thresh=15):
    res = np.ones_like(x)
    ind = np.abs(x) < thresh
    res[x < -thresh] = 0
    res[ind] =  1 / (1 + np.exp(-x[ind]))
    return res

act_numpy = {
    'soft_relu': soft_relu,
    'sigmoid': sigmoid,
    'sigmoid_stable': sigmoid_stable,
}

activation_templates = {
    'soft_relu': lambda _in, _out, _thr: """
    if( {0} < -{2} ) {{
        {1} = 0.0f;
    }} else if ( -{2} < {0} && {0} < {2} ) {{
        {1} = log( 1.0f + exp({0}));
    }} else {{
        {1} = {0};
    }}
    """.format(_in, _out, _thr),

    'sigmoid': lambda _in, _out, _thr: """
    {1} = 1.0f / (1.0f + exp(-{0}));
    """.format(_in, _out, _thr),

    'sigmoid_stable': lambda _in, _out, _thr: """
    {1} = 1.0;
    if( {0} < -{2} ) {{
        {1} = 0.0f;
    }} else if ( -{2} < {0} && {0} < {2} ) {{
        {1} = 1.0f / (1.0f + exp(-{0}));
    }}
    """.format(_in, _out, _thr),
}

# Network units behaviour, particularly useful for
# defining threshold code in neurons
threshold_templates = {
    "none": lambda comp_var, thr_var: """
        0
    """,  # this will block neurons' spiking
    "greater_than": lambda comp_var, thr_var: """
    {0} >= {1}
    """.format(comp_var, thr_var),
}

RunModes = RUN_MODES()

Stages = STAGES()

ObjectTypes = OBJECT_TYPES()

debug_prints = bool(0)


