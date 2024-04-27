from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# import tensorflow.contrib.slim as slim
import tf_slim as slim
# from tensorflow.contrib.layers import xavier_initializer
from pprint import pprint

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.model import Model
from ray.rllib.models.tf.misc import linear, normc_initializer
class CarlaImitationModel(Model):
    def _build_layers_v2(self, inputs, num_outputs, options):
        hiddens_vision = []
        hiddens_measure = [128,128,64]
        hiddens_concat_out = [512,256,256]
        activation = tf.nn.relu
        obs = inputs["obs"]
        vision_in = obs[0]
        metrics_in = obs[1]
        print("***Vision in shape", vision_in)
        print("***Vision in shape", metrics_in)

        # Setup metrics layer
        with tf.name_scope("carla_metrics"):
            i = 0
            for size in hiddens_measure:
                i += 1
                metrics_in = slim.fully_connected(
                    metrics_in,
                    size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
                    activation_fn=activation,
                    scope="metrics_out{}".format(i))
        vision_out = slim.flatten(vision_in)
        print("***Shape of vision out after flatten is", vision_out.shape)
        print("***Shape of metric out is", metrics_in.shape)

        # Combine the metrics and vision inputs
        with tf.name_scope("carla_out"):
            i = 1
            last_layer = tf.concat([vision_out, metrics_in], axis=1)
            # print("***Shape of concatenated out is", last_layer.shape)
            for size in hiddens_concat_out:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
                    activation_fn=activation,
                    scope="carla_out_fc{}".format(i))
                i += 1
            output = last_layer
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")
        return output, last_layer

def register_carla_imitation_model():
    print(ModelCatalog)
    print("type:", type(ModelCatalog))
    print(dir(ModelCatalog))
    ModelCatalog.register_custom_model("carla_imitation_model", CarlaImitationModel)

class CarlaModel(Model):
    def _build_layers_v2(self, inputs, num_outputs, options):
        # Parse options
        # image_shape = options["custom_options"].get("image_shape", [160, 320, 3])
        convs = [[32, [5, 5], 4], [64, [3, 3], 2], [128, [3, 3], 2], [256, [3, 3], 2]]
        # final layer output (4，9, 256)
        hiddens_vision = options.get("fcnet_hiddens", [256, 128])
        hiddens_measure = [128, 64]
        hiddens_concat_out = [64]
        fcnet_activation = options.get("fcnet_activation", "relu")
        activation = tf.nn.relu
        obs = inputs["obs"]
        vision_in = obs[0]
        metrics_in = obs[1]
        print("***Vision in shape", vision_in)
        '''
        >>> Vision in shape Tensor("car2/default_model_1/Reshape:0", shape=(?, x,y,channel), dtype=float32)
        >>> Metrics in shape Tensor("car2/default_model_1/Reshape_1:0", shape=(?, 5), dtype=float32)
        '''
        # Setup vision layers
        with tf.name_scope("carla_vision"):
            for i, (out_size, kernel, stride) in enumerate(convs, 1):
                vision_in = slim.conv2d(
                    vision_in,
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    padding="VALID",
                    scope="conv{}".format(i))
            print("***Shape of vision CNN out is", vision_in.shape)
            # final layer output (4,9,256)
            vision_out = slim.flatten(vision_in)
            i = 1
            for size in hiddens_vision:
                vision_out = slim.fully_connected(
                    vision_out,
                    size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
                    activation_fn=activation,
                    scope="vision_fc{}".format(i))
                i += 1
        # Setup metrics layer
        with tf.name_scope("carla_metrics"):
            i = 0
            for size in hiddens_measure:
                i += 1
                metrics_in = slim.fully_connected(
                    metrics_in,
                    size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
                    activation_fn=activation,
                    scope="metrics_out{}".format(i))
        print("***Shape of vision out after flatten is", vision_out.shape)  #(?,256)
        print("***Shape of metric out is", metrics_in.shape) #(?,64)

        # Combine the metrics and vision inputs
        with tf.name_scope("carla_out"):
            i = 1
            last_layer = tf.concat([vision_out, metrics_in], axis=1)
            # print("***Shape of concatenated out is", last_layer.shape)
            for size in hiddens_concat_out:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
                    activation_fn=activation,
                    scope="carla_out_fc{}".format(i))
                i += 1
            output = last_layer
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope="fc_out")

        return output, last_layer


'''
***Shape of vision out is 
***Shape of vision out after flatten is (?, )
***Shape of metric out is (?, )
***Shape of concatenated out is (?, )
'''


def register_carla_model():
    print(ModelCatalog)
    print("type:", type(ModelCatalog))
    print(dir(ModelCatalog))
    ModelCatalog.register_custom_model("carla", CarlaModel)


class CarlaModelv0(Model):
    """Carla model that can process the observation tuple.

    The architecture processes the image using convolutional layers, the
    metrics using fully connected layers, and then combines them with
    further fully connected layers.

    Define the layers of a custom model.

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

    def _build_layers_v2(self, inputs, num_outputs, options):
        # Parse options
        print("****numoutputs", num_outputs)
        image_shape = options["custom_options"].get("image_shape", [84, 84, 3])
        convs = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]]
        # layer3 output:(11, 11, 64)
        hiddens = options.get("fcnet_hiddens", [64])
        fcnet_activation = options.get("fcnet_activation", "tanh")
        activation = tf.nn.relu
        obs = inputs["obs"]
        vision_in = obs[0]
        metrics_in = obs[1]
        # print("***Vision in shape", vision_in)
        '''
        >>> Vision in shape Tensor("car2/default_model_1/Reshape:0", shape=(?, 84, 84, 3), dtype=float32)
        >>> Metrics in shape Tensor("car2/default_model_1/Reshape_1:0", shape=(?, 4), dtype=float32)
        '''
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
            vision_out = slim.conv2d(
                vision_in,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="conv_out")
        # print("***Shape of vision out is", vision_out.shape)
        vision_out = slim.flatten(vision_out)
        # Setup metrics layer
        with tf.name_scope("carla_metrics"):
            metrics_in = slim.fully_connected(
                metrics_in,
                64,
                weights_initializer=tf.keras.initializers.glorot_normal(),
                activation_fn=activation,
                scope="metrics_out1")

        # print("***Shape of vision out after flatten is", vision_out.shape)
        # print("***Shape of metric out is", metrics_in.shape)

        # Combine the metrics and vision inputs
        with tf.name_scope("carla_out"):
            i = 1
            last_layer = tf.concat([vision_out, metrics_in], axis=1)
            # print("***Shape of concatenated out is", last_layer.shape)
            for size in hiddens:
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=tf.keras.initializers.glorot_normal(),
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


'''
***Shape of vision out is (?, 9, 9, 64)
***Shape of vision out after flatten is (?, 5184)
***Shape of metric out is (?, 64)
***Shape of concatenated out is (?, 5248)
'''


def register_carla_modelv0():
    print(ModelCatalog)
    print("type:", type(ModelCatalog))
    print(dir(ModelCatalog))
    ModelCatalog.register_custom_model("carlav0", CarlaModelv0)


filters_mnih15 = [[32, [8, 8], 4], [64, [4, 4], 2], [64, [3, 3], 1]]


class Mnih15(Model):
    """
    Network definition as per Mnih15, Nature paper Methods section
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
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
