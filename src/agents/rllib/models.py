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
from ray.rllib.models.model import Model


class CarlaModel(Model):
    """Carla model that can process the observation tuple.

    The architecture processes the image using convolutional layers, the
    metrics using fully connected layers, and then combines them with
    further fully connected layers.
    """


    def _build_layers_v2(self, inputs, num_outputs, options):
        # Parse options
        image_shape = options["custom_options"]["image_shape"]
        convs = options.get("conv_filters", [
            [16, [8, 8], 4],
            [32, [5, 5], 3],
            [32, [5, 5], 2],
            [512, [10, 10], 1],
        ])
        hiddens = options.get("fcnet_hiddens", [64])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        if fcnet_activation == "tanh":
            activation = tf.nn.tanh
        elif fcnet_activation == "relu":
            activation = tf.nn.relu

        # Sanity checks
        image_size = np.product(image_shape)
        expected_shape = [image_size + 5 + 2]
        assert inputs.shape.as_list()[1:] == expected_shape, \
            (inputs.shape.as_list()[1:], expected_shape)

        # Reshape the input vector back into its components
        vision_in = tf.reshape(inputs[:, :image_size],
                               [tf.shape(inputs)[0]] + image_shape)
        metrics_in = inputs[:, image_size:]
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
                    activation_fn=activation,
                    padding="SAME",
                    scope="conv{}".format(i))
            out_size, kernel, stride = convs[-1]
            vision_in = slim.conv2d(
                vision_in,
                out_size,
                kernel,
                stride,
                padding="VALID",
                scope="conv_out")
            vision_in = tf.squeeze(vision_in, [1, 2])

        # Setup metrics layer
        with tf.name_scope("carla_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                64,
                weights_initializer=xavier_initializer(),
                activation_fn=activation,
                scope="metrics_out")

        print("Shape of vision out is", vision_in.shape)
        print("Shape of metric out is", metrics_in.shape)

        # Combine the metrics and vision inputs
        with tf.name_scope("carla_out"):
            i = 1
            last_layer = tf.concat([vision_in, metrics_in], axis=1)
            print("Shape of concatenated out is", last_layer.shape)
            for size in hiddens:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=xavier_initializer(),
                    activation_fn=activation,
                    scope="fc{}".format(i))
                i += 1
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")

        return output, last_layer


def register_carla_model():
    print(ModelCatalog)
    print("type:", type(ModelCatalog))
    print(dir(ModelCatalog))
    ModelCatalog.register_custom_model("carla", CarlaModel)


filters_mnih15 = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]]


class Mnih15(Model):
    """
    Network definition as per Mnih15, Nature paper Methods section
    """
    """Define the layers of a custom model.

    Arguments:
        input_dict (dict): Dictionary of input tensors, including "obs",
            "prev_action", "prev_reward", "is_training".
        num_outputs (int): Output tensor must be of size
            [BATCH_SIZE, num_outputs].
        options (dict): Model options.

    Returns:
        (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
            and [BATCH_SIZE, desired_feature_size].

    When using dict or tuple observation spaces, you can access
    the nested sub-observation batches here as well:

    Examples:
        >>> print(input_dict)
        {'prev_actions': <tf.Tensor shape=(?,) dtype=int64>,
            'prev_rewards': <tf.Tensor shape=(?,) dtype=float32>,
            'is_training': <tf.Tensor shape=(), dtype=bool>,
            'obs': OrderedDict([
            ('sensors', OrderedDict([
                ('front_cam', [
                    <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>,
                    <tf.Tensor shape=(?, 10, 10, 3) dtype=float32>]),
                ('position', <tf.Tensor shape=(?, 3) dtype=float32>),
                ('velocity', <tf.Tensor shape=(?, 3) dtype=float32>)]))])}
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        print("imput_dict:",input_dict)
        print("option", options)
        
        convs = options.get("conv_filters")
        if convs is None:
            convs = filters_mnih15
        activation = tf.nn.relu
        conv_output = input_dict["obs"]
        # pprint ("First")

        # pprint (conv_output)
        with tf.name_scope("mnih15_convs"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                conv_output = slim.conv2d(
                    input_dict["obs"],
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    padding="SAME",
                    scope="conv{}".format(i))
            # pprint ("Second")
            # pprint (conv_output)
            out_size, kernel, stride = convs[-1]
            conv_output = slim.conv2d(
                conv_output,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="conv_out")
            # pprint ("Third")
            # pprint (conv_output)
        action_out = slim.flatten(conv_output)
        with tf.name_scope("mnih15_FC"):
            shared_layer = slim.fully_connected(
                action_out, 128, activation_fn=activation)
            action_logits = slim.fully_connected(
                action_out, num_outputs=num_outputs, activation_fn=None)
        # pprint ("Fourth")
        # pprint (action_logits)
        return action_logits, shared_layer


def register_mnih15_net():
    ModelCatalog.register_custom_model("mnih15", Mnih15)


class Mnih15SharedWeights(Model):
    """
    Network definition as per Mnih15, Nature paper Methods section.
    Shared FC layers for use in Multi-Agent settings
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        convs = options.get("conv_filters")
        if convs is None:
            convs = filters_mnih15
        activation = tf.nn.relu
        conv_output = input_dict["obs"]
        with tf.name_scope("mnih15_convs"):
            for i, (out_size, kernel, stride) in enumerate(convs[:-1], 1):
                conv_output = slim.conv2d(
                    input_dict["obs"],
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    padding="SAME",
                    scope="conv{}".format(i))
            out_size, kernel, stride = convs[-1]
            conv_output = slim.conv2d(
                conv_output,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="conv_out")
        action_out = slim.flatten(conv_output)

        with tf.name_scope("mnih15_FC"):
            # Share weights of the following layer with other instances of this
            # model (usually by other macad_agents in a Multi-Agent setting)
            with tf.variable_scope(
                    tf.VariableScope(tf.AUTO_REUSE, "shared"),
                    reuse=tf.AUTO_REUSE):
                shared_layer = slim.fully_connected(
                    action_out, 128, activation_fn=activation)
            action_logits = slim.fully_connected(
                action_out, num_outputs=num_outputs, activation_fn=None)
        return action_logits, shared_layer


def register_mnih15_shared_weights_net():
    ModelCatalog.register_custom_model("mnih15_shared_weights",
                                       Mnih15SharedWeights)