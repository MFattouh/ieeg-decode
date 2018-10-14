from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.HYBRID = dict()
# Spatial filters configurations
__C.HYBRID.SPATIAL_CONVS = dict()
__C.HYBRID.SPATIAL_CONVS.ENABLED = False
__C.HYBRID.SPATIAL_CONVS.num_filters = []  # number of output filters for each layer
__C.HYBRID.SPATIAL_CONVS.kernel_size = []  # kernel size for each layer
__C.HYBRID.SPATIAL_CONVS.batch_norm = False
__C.HYBRID.SPATIAL_CONVS.activations = 'tanh'

# Temporal filters configurations
__C.HYBRID.TEMPORAL_CONVS = dict()
__C.HYBRID.TEMPORAL_CONVS.ENABLED = False
__C.HYBRID.TEMPORAL_CONVS.num_filters = []  # number of output filters for each layer
__C.HYBRID.TEMPORAL_CONVS.batch_norm = False
__C.HYBRID.TEMPORAL_CONVS.activations = 'tanh'

# RNN configs
__C.HYBRID.RNN = dict()
__C.HYBRID.RNN.rnn_type = 'gru'
__C.HYBRID.RNN.normalization = None  # 'batch_norm' is supported, 'weight_norm' will be supported soon
__C.HYBRID.RNN.dropout = 0.3  # or a list of values
__C.HYBRID.RNN.weights_dropout = ['weight_ih_l0', 'weight_ih_l1', 'weight_ih_l2', 'weight_hh_l0', 'weight_hh_l1', 'weight_hh_l2']
__C.HYBRID.RNN.max_length = 0  # maximum sequence length (required for BNLSTM)
__C.HYBRID.RNN.hidden_size = 64
__C.HYBRID.RNN.num_layers = 3

# L2POOLING
__C.HYBRID.L2POOLING = dict()
__C.HYBRID.L2POOLING.ENABLED = False
__C.HYBRID.L2POOLING.window = 10
__C.HYBRID.L2POOLING.stride = 1

__C.HYBRID.LINEAR = dict()
__C.HYBRID.LINEAR.batch_norm = False
__C.HYBRID.LINEAR.dropout = [0.5, 0.3]
__C.HYBRID.LINEAR.fc_size = [32, 10]
__C.HYBRID.LINEAR.activations = 'tanh'

__C.TRAINING = dict()
__C.TRAINING.RANDOM_SEED = 10418

__C.OPTIMIZATION = dict()
__C.OPTIMIZATION.OPTIMIZER = 'adam'
__C.OPTIMIZATION.BASE_LR = 5e-3
__C.OPTIMIZATION.WEIGHT_DECAY = 5e-6


def _merg_a_into_b(a, b):
    assert isinstance(a, (edict, dict))
    assert isinstance(b, (edict, dict))

    for k, v in a.items():
        try:
            if isinstance(v, (edict, dict)):
                _merg_a_into_b(a[k], b[k])
            else:
                b[k] = _decode_value(v)
        except KeyError as kerror:
            raise KeyError(kerror)


def _decode_value(value):
    """
    Decodes values from yaml parser.

    It tries to convert the value from str to list, tuple, ints and floats,
    in that order.

    **Nested lists and tuples will be merged into a single container**
    :param value: value to decode
    :return: decoded value
    """
    if not isinstance(value, str):
        return value
    elif value.startswith('['):
        return list(map(_decode_value, value.strip('[] ').split(',')))
    elif value.startswith('('):
        value = tuple(map(_decode_value, value.strip('() ').split(',')))
        return value
    else:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def merge_configs(other_configs):
    """
    Merge configurateions in other_configs to the global conf
    :param other_configs: dict of configurations
    :return: None
    """
    assert isinstance(other_configs, (edict, dict))
    _merg_a_into_b(other_configs, __C)
