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

"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf
import numpy as np


class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.

  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
          all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class SoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 10e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)

class MultiLabelCrossEntropy(BaseLoss):
  """Basically delegates to tf.nn.sigmoid_cross_entropy_with_logits.
  Loss should be a measure of how far the actual distribution is from
  the uniform distribution of the given labels.
  Possible that we want to do a TF IDF here to scale the labels.  Hmm.
  We'd want to give more loss if we give high prediction to rare classes.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 10e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)


class MultilabelLearningLoss(BaseLoss):
  """Calculate the Multi-label loss between the predictions and the labels.
  Loss considers correlation between similar labels, and penalizes for 
  ranking of labels, not just presence.

  From Zhang and Zhou, TKDE 2005

  A reminder for me:

     Args:
      predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      labels: a 2-d tensor storing the labels, which has the same shape
        as the predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.
  """

  # broadcasting matmul courtesy of https://github.com/tensorflow/tensorflow/issues/216
  # Nevermind, this just flat out does nothing and breaks everything.  Thanks.
  def batch_vm(self, v, m):
    shape = tf.shape(v)
    rank = shape.get_shape()[0].value
    v = tf.expand_dims(v, rank)

    vm = tf.matmul(v, m)

    return tf.reduce_sum(vm, rank-1)
 
  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_mll"):

      # epsilon = 10e-6

      # debug
      # labels = tf.constant([[1, 0, 1]])
      # predictions = tf.constant([[0.7, 0.2, 0.1]], dtype=tf.float32)

      float_labels = tf.cast(labels, tf.float32)
      # Get the number of tags per frame
      # num_tags = tf.reduce_sum(labels, 1)

      #naive way of doing this.  Argh.  Need to do this faster, matrix math?
      # let's do this really naively
      # # (num_rows,num_classes) = tf.shape(predictions)
      # np_predictions = np.array(tf.unstack(predictions))
      # np_labels = np.array(tf.unstack(labels))

      # (num_rows, num_classes) = np_predictions.shape
      # assert((num_rows,num_classes) == np_labels.shape)
      # ml_loss = 0.0
      # for i in num_rows:
      #   row_loss = 0.0
      #   num_tags = 0
      #   for j in num_classes:
      #     if np_labels[i,j] == 1:
      #       num_tags += 1
      #       for k in num_classes:
      #         if np_labels[i,k] == 0:
      #           row_loss += np.exp(-(np_predictions[i,j] - np_predictions[i,k]))
      #   row_loss /= (num_tags * (num_classes - num_tags))
      #   ml_loss += row_loss

      # Looks like Tensorflow needs us to do this as tensor math
      [num_rows, num_classes] = predictions.get_shape().as_list()

      # tensorflow won't give me the shape in this way, num_rows comes back as a ?
      # So for now, hard set batch size at 128
      # ACTUALLY - computationally, we have to set this batch size to 1.
      # num_rows = 128
      num_rows = 1

      #r_pred = tf.reshape(predictions, [num_rows, num_classes, 1])
      # debug
      # broadcaster = tf.ones([1, 3], dtype=tf.float32)
      broadcaster = tf.ones([1, num_classes], dtype=tf.float32)

      # cross differences, need to take difference from cross product, then exponentiate, then sum along
      # rows and divide by |y_i| |\hat{y_i}| per row
      cross_predictions = tf.matmul(tf.reshape(predictions, [num_rows*num_classes, 1]), broadcaster)
      cross_differences = tf.subtract(cross_predictions, tf.transpose(cross_predictions))
      cross_differences = tf.exp(tf.multiply(cross_differences, tf.constant(-1.0)))

      # cross_label_mask is the xor of cross product of row labels
      # xor can be adding together and then mod by 2
      # cross_label_mask = tf.mod(tf.add(tf.matmul(tf.reshape(float_labels, [num_rows, num_classes, 1]), broadcaster), tf.matmul(tf.reshape(float_labels, [num_rows, 1, num_classes]), tf.transpose(broadcaster))), tf.constant(2))
      # float_labels = tf.Print(float_labels, [float_labels], message="float_labels is : ")
      # label_sum = tf.reduce_sum(float_labels)
      # label_shape = tf.Print(label_shape, [label_shape], message="l is : ")
      # label_sum = tf.Print(label_sum, [label_sum], message="label_sum is : ")
      # cross_label_mask = tf.mod(tf.add(tf.matmul(tf.reshape(float_labels, [num_rows*num_classes, 1]), broadcaster), tf.transpose(tf.matmul(tf.transpose(broadcaster), tf.reshape(float_labels, [1, num_rows*num_classes])))), 2.0)
      broadcasted_labels = tf.matmul(tf.reshape(float_labels, [num_rows*num_classes, 1]), broadcaster)

      cross_label_mask = tf.mod(tf.add(broadcasted_labels, tf.transpose(broadcasted_labels)), 2.0)
      # We actually only want the upper triangular part of this, since it is symmetric and is actually
      # redundant to count each pair twice - they should not be ordered pairs
      # TF special case to take upper triangular
      cross_label_mask = tf.matrix_band_part(cross_label_mask, 0, -1)
      # cross_sum = tf.reduce_sum(cross_label_mask, [0,1])
      # cross_sum = tf.Print(cross_sum, [cross_sum], message="cross_sum is : ")
      # cross_label_shape = tf.shape(cross_label_mask)
      # print("HHHHHHHHHHHHHHHHHH HHHHHHHHHHHHHHHHHH HHHHHHHHHHHHHHHHHH " + str(cross_label_mask.get_shape()))

      # cross_differences is (N,C,C)
      # cross_label_mask is (N,C,C)
      # want to apply mask
      subset_sum = tf.multiply(cross_differences, cross_label_mask)
      subset_sum = tf.reshape(subset_sum, [num_rows, num_classes, num_classes])
      cross_label_mask = tf.reshape(cross_label_mask, [num_rows, num_classes, num_classes])
      reduced_sum = tf.reduce_sum(subset_sum)

      # We scale the loss of each instance by the number of probabilities that contributed
      # to its loss
      instance_scale = tf.reduce_sum(cross_label_mask, [1,2])
      # instance_scale = tf.Print(instance_scale, [instance_scale])
      instance_scale = tf.expand_dims(instance_scale, -1)
      instance_scale = tf.expand_dims(instance_scale, -1)
      # instance_scale = tf.Print(instance_scale, [cross_differences, cross_label_mask, reduced_sum, instance_scale], message="cross_differences, cross_label_mask, reduced_sum, instance_scale is : ", summarize=10)
      # tensorflow?????? why would you break the convention of literally every other
      # data analysis tool that has ever been written?
      # courtesy of http://stackoverflow.com/a/36296783/965600
      # Add print operation
#      a = tf.Print(instance_scale, [instance_scale], message="This is a: ")

      # Add more elements of the graph using a
#      b = tf.add(a, a).eval()

      # ml_loss = tf.reduce_sum(tf.divide(subset_sum, instance_scale))
      ml_loss = tf.reduce_sum(subset_sum / instance_scale)
      # return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))
      # ml_loss = tf.Print(ml_loss, [ml_loss], message="ml loss should be: ")
      return ml_loss

