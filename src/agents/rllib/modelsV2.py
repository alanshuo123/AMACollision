from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim
#from tensorflow.contrib.layers import xavier_initializer
from pprint import pprint

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

filters_mnih15 = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]]

class CarlaModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CarlaModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # Parse options
        image_shape = model_config["custom_options"].get("image_shape", [84,84,3])
        convs = model_config.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [5, 5], 3],
            [32, [5, 5], 2],
            [512, [10, 10], 1],
        ])
        hiddens = model_config.get("fcnet_hiddens", [64])
        fcnet_activation = model_config.get("fcnet_activation", "relu")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        # Define the input tensors
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="imputs")

        # Reshape the input vector back into its components
        vision_in = tf.reshape(
            self.inputs[:, :np.product(image_shape)], [-1] + list(image_shape))
        metrics_in = self.inputs[:, np.product(image_shape):]
        print("Vision in shape", vision_in)
        print("Metrics in shape", metrics_in)

        # Setup vision layers
        with tf.name_scope("carla_vision"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                vision_in = slim.conv2d(
                    vision_in, 
                    out_size, 
                    kernel, 
                    stride,
                    scope="conv{}".format(i))
            out_size, kernel, stride = convs[-1]
            vision_in = slim.conv2d(
                vision_in, 
                out_size, 
                kernel, 
                stride,
                padding="VALID", scope="conv_out")
            vision_in = tf.squeeze(vision_in, [1, 2])

        # Setup metrics layer
        with tf.name_scope("carla_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in, 
                64,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                activation_fn=activation,
                scope="metrics_out")
        print("Shape of vision out is", vision_in.shape)
        print("Shape of metric out is", metrics_in.shape)

        # Combine the metrics and vision inputs
        with tf.name_scope("carla_out"):
            last_layer = tf.concat([vision_in, metrics_in], axis=1)
            print("Shape of concatenated out is", last_layer.shape)
            for i, size in enumerate(hiddens, 1):
                last_layer = slim.fully_connected(
                    last_layer, size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
                    activation_fn=activation,
                    scope="fc{}".format(i))
            self.action_out = slim.fully_connected(
                last_layer, 
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None, 
                scope="action_out")

        self.value_out = slim.fully_connected(
            last_layer,
            1,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="value_out"
        )
        self.base_model = tf.keras.Model(self.inputs, [self.action_out, self.value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(
            tf.cast(input_dict["obs"], tf.float32))

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class Mnih15(TFModelV2):
    """
    Network definition as per Mnih15, Nature paper Methods section
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(Mnih15, self).__init__(obs_space, action_space,
                                            num_outputs, model_config, name)

        activation = tf.nn.relu
        convs = model_config.get("conv_filters")
        if convs is None:
            convs = filters_mnih15

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        conv_output = inputs

        with tf.name_scope("mnih15_convs"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                conv_output = tf.keras.layers.Conv2D(
                    out_size, 
                    kernel, 
                    strides=stride, 
                    activation=activation, 
                    padding="same", 
                    name="conv{}".format(i))(conv_output)

            out_size, kernel, stride = convs[-1]
            conv_output = tf.keras.layers.Conv2D(
                out_size, 
                kernel, 
                strides=stride, 
                activation=activation, 
                padding="valid", 
                name="conv_out")(conv_output)

        action_out = tf.keras.layers.Flatten()(conv_output)

        with tf.name_scope("mnih15_FC"):
            shared_layer = tf.keras.layers.Dense(128, activation=activation)(action_out)
            action_logits = tf.keras.layers.Dense(num_outputs, activation=None)(action_out)

        self.base_model = tf.keras.Model(inputs, [action_logits, shared_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

def register_mnih15_net():
    ModelCatalog.register_custom_model("mnih15", Mnih15)