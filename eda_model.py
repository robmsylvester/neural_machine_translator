# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_utils
import vocabulary_utils

FLAGS = tf.app.flags.FLAGS


class EncoderDecoderAttentionModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               max_clipped_gradient,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               min_learning_rate,
               use_lstm=True,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.global_step = tf.Variable(0, trainable=False)
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.minimum_learning_rate = tf.constant(min_learning_rate, dtype=tf.float32)
    self.learning_rate_decay_factor = tf.constant(learning_rate_decay_factor, dtype=tf.float32)


    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=True, dtype=dtype)

    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.maximum(self.minimum_learning_rate, self.learning_rate * self.learning_rate_decay_factor, "learning_rate_decay"))



    # If we use sampled softmax, we need an output projection.
    #output_projection = None
    softmax_loss_function = None

    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:

      output_projection = model_utils._create_output_projection(self.target_vocab_size,
                                                                            FLAGS.decoder_hidden_size)
      weights = output_projection[0]
      biases = output_projection[1]
      weights_t = output_projection[2]

      #weights_t = tf.get_variable("output_projection_weights", [self.target_vocab_size, FLAGS.decoder_hidden_size], dtype=dtype)
      #weights = tf.transpose(weights_t)
      #biases = tf.get_variable("output_projection_biases", [self.target_vocab_size], dtype=dtype)
      #output_projection = (weights, biases)

      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        #local_w_t = tf.cast(w_t, tf.float32)
        #local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=weights_t,
                biases=biases,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return model_utils.run_model(
          encoder_inputs,
          decoder_inputs,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=FLAGS.encoder_embedding_size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []

    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
      
    #We add one to the buckets for the decoder size because we add a symbol.
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = model_utils.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)

      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, weights) + biases
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = model_utils.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_clipped_gradient)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id, load_from_memory=True, source_path=None, target_path=None, max_size=None):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
      load_from_memory : boolean - if true, loads the dataset from memory by
        directly examining the buckets and randomly choosing from them. if false,
        will randomly choose lines fitting the sizing parameters of the bucket_id
        that was passed. notice that this will be slower.

      #TODO - finish implementing getting a batch from not the file

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """

    if load_from_memory:
      return self.get_batch_from_memory(data, bucket_id)
    else:
      return self.get_batch_from_file(data, bucket_id, source_path, target_path, max_size)



  def clean_encoder_and_decoder_inputs(self, data, bucket_id):
    """Cleaning encoder and decoder inputs amounts to padding the inputs to the size of the bucket for
    both the encoder and decoder, as well as adding a go symbol to the beginning of the decoder sentence.
    This function will also reverse the input encoder sentence"""
    encoder_inputs = []
    decoder_inputs = []

    encoder_size, decoder_size = self.buckets[bucket_id]

    # Get a random batch of encoder and decoder inputs from data
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded so they fill up the bucket length
      encoder_pad = [vocabulary_utils.PAD_ID] * (encoder_size - len(encoder_input))

      # Then we reverse the inputs as per Cho's recommendations
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol since we feed previous outputs to a decoder, and this will handle that base case
      decoder_pad_size = decoder_size - len(decoder_input) - 1

      #And we pad the decoder inputs to fill the rest of the bucket as well
      decoder_inputs.append([vocabulary_utils.GO_ID] + decoder_input +
                            [vocabulary_utils.PAD_ID] * decoder_pad_size)

    return encoder_size, decoder_size, encoder_inputs, decoder_inputs


  def weight_target_symbols(self, decoder_inputs, decoder_size, length_idx):
    """As it stands now in this implementation, we don't really care about learning the padding term
    at all, so we zero it out in the loss function calculation. We also zero out the final symbol of
    the decoder as well, namely the end of the sentence. Every other target gets an equal weight of
    1.0. These weights will be multiplied by their corresponding decoder indexes in the loss function
    when the overall softmax cross-entropy is calculated. In time, it might be worthwhile to dynamically
    adjust the target weights if we find that we are misclassifying certain labels. One way to do this would
    be to examine the perplexity of a sentence and boost the weights of those particular labels based
    on how well we do on that sentence.

    For example, if a sentence "I have a dog" translated to "Io non ho un gatto" (I don't have a cat), then
    perhaps we need to boost the importance of getting the words "I", "have", "a" and "dog". If we nailed
    the sentence, and it therefore has low perplexity, the weights for these logits could decrease

    For now, we simply treat PAD_ID as a weight of 0, and we treat the final length index as 0. all other
    words get an importance of 1."""


    batch_weight = np.ones(self.batch_size, dtype=np.float32)

    for batch_idx in xrange(self.batch_size):
      if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
      if length_idx == decoder_size - 1 or target == vocabulary_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
    return batch_weight



  def get_batch_from_memory(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
      load_from_memory : boolean - if true, loads the dataset from memory by
        directly examining the buckets and randomly choosing from them. if false,
        will randomly choose lines fitting the sizing parameters of the bucket_id
        that was passed. notice that this will be slower.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """

    encoder_size, decoder_size, encoder_inputs, decoder_inputs = self.clean_encoder_and_decoder_inputs(data, bucket_id)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = []
    batch_decoder_inputs = []
    batch_weights = []

    #our inputs need to be reshaped so that we feed in the symbol at time step t for each example in the batch
    #encoder inputs are a list, so make an array out of the list and take the tranpose of the matrix to do this.
    encoder_input_as_array = np.transpose(np.stack(encoder_inputs, axis=0)).astype(np.int32)
    decoder_input_as_array = np.transpose(np.stack(decoder_inputs, axis=0)).astype(np.int32)

    #fetch a coefficient that corresponds to how much we care about getting each decoder input correct
    batch_weights = [self.weight_target_symbols(decoder_inputs, decoder_size, l_idx) for l_idx in xrange(decoder_size)]

    #split the array back into a list of symbols, which creates a list of [1,x] shapes, so we squeeze them into [x,] shapes
    #in the event we have a batch size of 1, which means we are probably using the decoder, then squeeze would accidentally squeeze
    #out an extra dimension, so we don't want that, hence we pass an axis=0
    batch_encoder_inputs = [np.squeeze(i, axis=0) for i in np.split(encoder_input_as_array, encoder_size, axis=0)]
    batch_decoder_inputs = [np.squeeze(i, axis=0) for i in np.split(decoder_input_as_array, decoder_size, axis=0)]
    end2 = time.time()

    #make sure they all match
    #for idx in xrange(encoder_size):
    #  assert(np.array_equal(batch_encoder_inputs[idx], batch_encoder_inputs2[idx])), "implementations encoder dont match"
    #  assert(np.array_equal(batch_decoder_inputs[idx], batch_decoder_inputs2[idx])), "implementations decoder dont match"
    #  assert(np.array_equal(batch_weights[idx], batch_weights2[idx])), "implementations weights dont match"

    #assert len(batch_encoder_inputs) == len(batch_encoder_inputs2), "encoder lengths dont match"
    #assert len(batch_decoder_inputs) == len(batch_decoder_inputs2), "decoder lengths dont match"
    #assert len(batch_weights) == len(batch_weights2), "encoder lengths dont match"

    #decoder_input_copy = np.copy(decoder_inputs)
    #decoder_input_copy = np.transpose(decoder_input_copy, perm=[1,0,2])

    #raise ValueError("Intentionally stopping. Everything works!")

    #print(batch_encoder_inputs[0].shape)
    #print(len(batch_encoder_inputs))
    #print(batch_decoder_inputs[0].shape)
    #print(len(batch_decoder_inputs))
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


  def get_batch_from_file(self, data, bucket_id, source_path, target_path, max_size):
    raise NotImplementedError
