# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools

import tensorflow as tf

from tfimm.models.config import ModelConfig


def keras_serializable(cls):
    """
    Decorate a Keras Layer class to support Keras serialization.

    This is done by:

    1. Adding a `cfg` dict to the Keras config dictionary in `get_config` (called by
       Keras at serialization time.
    2. Wrapping `__init__` to accept that `cfg` dict (passed by Keras at deserialization
       time) and convert it to a config object for the actual layer initializer.
    3. Registering the class as a custom object in Keras (if the Tensorflow version
       supports this), so that it does not need to be supplied in `custom_objects` in
       the call to `tf.keras.models.load_model`.

    Args:
        cls (a `tf.keras.layers.Layer` subclass):
            Typically the main network class in this project, in general must accept a
            `cfg` argument to its initializer.

    Returns:
        The same class object, with modifications for Keras serialization.
    """
    initializer = cls.__init__

    cfg_class = getattr(cls, "cfg_class", None)
    if cfg_class is None:
        raise AttributeError("Must set `cfg_class` to use @keras_serializable")

    @functools.wraps(initializer)
    def wrapped_init(self, *args, **kwargs):
        cfg = (
            args[0]
            if args and isinstance(args[0], ModelConfig)
            else kwargs.pop("cfg", None)
        )

        if isinstance(cfg, dict):
            cfg = cfg_class.from_dict(cfg)
            initializer(self, cfg, *args, **kwargs)
        elif isinstance(cfg, ModelConfig):
            if len(args) > 0:
                initializer(self, *args, **kwargs)
            else:
                initializer(self, cfg, *args, **kwargs)
        else:
            raise ValueError("Must pass either `cfg` (ModelConfig) or `cfg` (dict)")

        self._cfg = cfg
        self._kwargs = kwargs

    cls.__init__ = wrapped_init

    if not hasattr(cls, "get_config"):
        raise TypeError(
            "Only use @keras_serializable on tf.keras.layers.Layer subclasses"
        )
    if hasattr(cls.get_config, "_is_default"):

        def get_config(self):
            _cfg = super(cls, self).get_config()
            _cfg["cfg"] = self._cfg.to_dict()
            _cfg.update(self._kwargs)
            return _cfg

        cls.get_config = get_config

    cls._keras_serializable = True
    if hasattr(tf.keras.utils, "register_keras_serializable"):
        cls = tf.keras.utils.register_keras_serializable()(cls)
    return cls
