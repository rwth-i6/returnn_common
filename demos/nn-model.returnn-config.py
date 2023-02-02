#!returnn.py
"""
Use this file as a RETURNN config.

Here we demonstrate how to define a simple model and how to setup training.
Also see: https://github.com/rwth-i6/returnn_common/wiki/RETURNN-example-config

This config is similar as the pure-RETURNN demo-tf-native-lstm.12ax.config.

Run as::

  returnn/rnn.py returnn_common/demos/nn-model.returnn-config.py

"""

import sys
from typing import Any, Dict
import os
from returnn.util.basic import get_login_username
from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

_my_dir = os.path.dirname(os.path.abspath(__file__))
_parent_rc_dir = os.path.dirname(os.path.dirname(_my_dir))
# Make sure we can import returnn_common.
# This is somewhat specific to this particular demo
# and will probably look different for you.
if _parent_rc_dir not in sys.path:
    sys.path.insert(0, _parent_rc_dir)

demo_name, _ = os.path.splitext(__file__)
print("Hello, experiment: %s" % demo_name)

use_tensorflow = True

task = "train"
train = {"class": "Task12AXDataset", "num_seqs": 1000}
dev = {"class": "Task12AXDataset", "num_seqs": 100, "fixed_random_seed": 1}

time_dim = SpatialDim("time")
feature_dim = FeatureDim("input", 9)
classes_dim = FeatureDim("classes", 2)
default_input = "data"
target = "classes"
extern_data = {
    "data": {"dim_tags": [batch_dim, time_dim, feature_dim]},
    "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim},
}


# model / network
def get_network(*, epoch: int, **_kwargs_unused) -> Dict[str, Any]:
    """called from the RETURNN config"""
    epoch  # noqa  # unused
    from returnn_common import nn

    nn.reset_default_root_name_ctx()
    data = nn.Data(name=default_input, **extern_data[default_input])
    targets = nn.Data(name=target, **extern_data[target])
    data = nn.get_extern_data(data)
    targets = nn.get_extern_data(targets)

    # We define a simple LSTM network.
    # This is similar as the pure-RETURNN demo-tf-native-lstm.12ax.config.
    class Model(nn.Module):
        """LSTM"""

        def __init__(self):
            super().__init__()
            hidden_dim = nn.FeatureDim("hidden", 10)
            self.lstm = nn.LSTM(feature_dim, hidden_dim)
            self.projection = nn.Linear(hidden_dim, classes_dim)

        def __call__(self, x: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
            x = nn.dropout(x, dropout=0.1, axis=feature_dim)
            x, _ = self.lstm(x, spatial_dim=spatial_dim)
            x = self.projection(x)
            return x  # logits

    net = Model()
    logits = net(data, spatial_dim=time_dim)
    loss = nn.sparse_softmax_cross_entropy_with_logits(logits=logits, targets=targets, axis=classes_dim)
    loss.mark_as_loss("ce")

    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=net)
    from returnn_common.utils.pprint import pprint

    pprint(net_dict)  # just for logging purpose
    return net_dict


# batching
batching = "random"
batch_size = 5000
max_seqs = 10
chunking = "200:200"

# training
optimizer = {"class": "adam"}
learning_rate = 0.01
# https://github.com/tensorflow/tensorflow/issues/6537
model = "/tmp/%s/returnn/%s/model" % (get_login_username(), demo_name)
start_epoch = 1  # ignore previous models, just for the demo
num_epochs = 5

# log
log_verbosity = 3
