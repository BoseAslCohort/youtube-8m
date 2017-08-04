# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])

    return {"predictions": final_probabilities}








"""It's a secret."""
class IsaacNet(models.BaseModel):


  def bottleneck(self, _input, out_features):
    with tf.variable_scope("bottleneck"):
        output = self.batch_norm(_input)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = self.conv2d(
            output, out_features=inter_features, kernel_size=1,
            padding='VALID')
        # TEMP TEMP TEMP
        #output = self.dropout(output)
    return output

  def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())
  def add_internal_layer(self, _input, growth_rate):
    """Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    if not self.bc_mode:
        comp_out = self.composite_function(
            _input, out_features=growth_rate, kernel_size=3)
    elif self.bc_mode:
        bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
        comp_out = self.composite_function(
            bottleneck_out, out_features=growth_rate, kernel_size=3)
    # concatenate _input with out from composite function
    output = tf.concat(axis=3, values=(_input, comp_out))
    return output

  def conv2d(self, _input, out_features, kernel_size,
             strides=[1, 1, 1, 1], padding='SAME'):
      in_features = int(_input.get_shape()[-1])
      kernel = self.weight_variable_msra(
          [kernel_size, kernel_size, in_features, out_features],
          name='kernel')
      output = tf.nn.conv2d(_input, kernel, strides, padding)
      return output

  def add_block(self, _input, growth_rate, layers_per_block):
      """Add N H_l internal layers"""
      output = _input
      for layer in range(layers_per_block):
          with tf.variable_scope("layer_%d" % layer):
              output = self.add_internal_layer(output, growth_rate)
      return output

  def batch_norm(self, _input):
    output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=self.is_training,
        updates_collections=None)
    return output

  def composite_function(self, _input, out_features, kernel_size=3):
      """Function from paper H_l that performs:
      - batch normalization
      - ReLU nonlinearity
      - convolution with required kernel
      - dropout, if required
      """
      with tf.variable_scope("composite_function"):
          # BN
          output = self.batch_norm(_input)
          # ReLU
          output = tf.nn.relu(output)
          # convolution
          output = self.conv2d(
              output, out_features=out_features, kernel_size=kernel_size)
          # dropout(in case of training and in case it is no 1.0)
          # todo TEMP TEMP TEMP
          #output = self.dropout(output)
      return output

  def transition_layer(self, _input):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * self.reduction)
    output = self.composite_function(
        _input, out_features=out_features, kernel_size=1)
    # run average pooling
    output = self.avg_pool(output, k=2)
    return output

  def avg_pool(self, _input, k):
      ksize = [1, k, k, 1]
      strides = [1, k, k, 1]
      padding = 'VALID'
      output = tf.nn.avg_pool(_input, ksize, strides, padding)
      return output

  def dropout(self, _input):
      if self.keep_prob < 1:
          output = tf.cond(
              self.is_training,
              lambda: tf.nn.dropout(_input, self.keep_prob),
              lambda: _input
          )
      else:
          output = _input
      return output

  def weight_variable_xavier(self, shape, name):
    return tf.get_variable(
        name,
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())

  def trainsition_layer_to_classes(self, _input):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    # BN
    output = self.batch_norm(_input)
    # ReLU
    output = tf.nn.relu(output)
    # average pooling
    last_pool_kernel = int(output.get_shape()[-2])
    output = self.avg_pool(output, k=last_pool_kernel)
    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    W = self.weight_variable_xavier(
        [features_total, self.n_classes], name='W')
    bias = self.bias_variable([self.n_classes])
    logits = tf.matmul(output, W) + bias
    return logits


  def bias_variable(self, shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """

    IsaacNet Copyright 2017

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """

    self.is_training = True
    self.keep_prob = 0.5
    self.reduction = 1.0
    self.bc_mode = True

    depth = 80
    total_blocks = 3
    growth_rate = 20

    self.n_classes = vocab_size

    layers_per_block = (depth - (total_blocks + 1)) // total_blocks
    first_output_features = growth_rate * 2


    #adj_input = tf.expand_dims(tf.expand_dims(model_input, 1), 1)
    #adj_input = tf.expand_dims(tf.reshape(model_input, [32, 32]), 1)
    adj_input = tf.reshape(model_input, [-1, 32, 32, 1])

    # first - initial 3 x 3 conv to first_output_features
    with tf.variable_scope("Initial_convolution"):
        output = self.conv2d(
            adj_input,
            out_features=first_output_features,
            kernel_size=3)

    # add N required blocks
    for block in range(total_blocks):
        with tf.variable_scope("Block_%d" % block):
            output = self.add_block(output, growth_rate, layers_per_block)
        # last block exists without transition layer
        if block != total_blocks - 1:
            with tf.variable_scope("Transition_after_block_%d" % block):
                output = self.transition_layer(output)

    output = tf.nn.dropout(output, .8)

    with tf.variable_scope("Transition_to_classes"):
        logits = self.trainsition_layer_to_classes(output)

    final_probabilities = tf.nn.softmax(logits)

    #final_probabilities = tf.Print(final_probabilities, [final_probabilities])


    return {"predictions": final_probabilities}
