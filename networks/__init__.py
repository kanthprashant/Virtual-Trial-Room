from __future__ import absolute_import

from networks.AugmentCE2P import resnet101
from networks.sdafnet import SDAFNet_Tryon

__factory = {
    'resnet101': resnet101,
    'sdafnet': SDAFNet_Tryon
}


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model arch: {}".format(name))
    return __factory[name](*args, **kwargs)
