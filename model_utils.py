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
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

#from tensorflow.contrib.rnn.python.ops import core_rnn
import encoder
import attention_decoder
import json
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS

def _create_output_projection(target_size,
                              output_size):
  #NOTE, shape[1] of the output projection weights should be equal to the output size of the final decoder layer
  #If, for some reason, I make the final output layer bidirectional or something else strange, this would be the
  #wrong size for the output

  #Notice we don't put this in a scope. It lives in the sequence model itself and gets passed between layers of the
  #decoder. We return a transpose variable itself to speed up training so we don't have to do this back and
  #forth between iterations. the weights transpose is necessary to pass to softmax. the weights themselves are used
  #to speak to the attention decoder.
  weights_t = tf.get_variable("output_projection_weights", [target_size, output_size], dtype=tf.float32)
  weights = tf.transpose(weights_t)
  biases = tf.get_variable("output_projection_biases", [target_size], dtype=tf.float32)
  return (weights, biases, weights_t)

def _verify_recurrent_stack_architecture(stack_json):
  permitted_input_merge_modes = [False,'concat','sum'] #when multiple layers connect to an LSTM/GRU, what do we do with these inputs? concat or sum, for now
  permitted_unidirectional_output_merge_modes = [False] #when a unidirectional LSTM has outputs for each time step, we dont have anything special to do. This is just for readability
  permitted_bidirectional_output_merge_modes = [False,'concat','sum']

  cur_layer_index = 0
  output_sizes = {}

  # example structure for what output sizes looks like
  # output_sizes = {
  #   'encoder0' : [512,512],
  #   'encoder1' : [512,512],
  #   'encoder2' : [1024],
  #   'encoder3' : [1024]
  # }
  
  #go one-by-one and make sure the architecture layer sizes add up
  for layer_name, layer_parameters in stack_json["layers"].iteritems():

    if cur_layer_index == 0:
      assert layer_parameters["input_merge_mode"] == False, "Input merge mode for first layer must be False"
    
    #merge mode is either concat or sum
    assert layer_parameters["input_merge_mode"] in permitted_input_merge_modes, "Merge mode in %s is invalid" % layer_name

    #no peephole connections on GRU's
    if not layer_parameters["lstm"]:
      assert not layer_parameters["peepholes"], "Cannot use peephole connections in layer %s because this is not an LSTM" % layer_name

    #Forget bias and dropout probabilities are in 0-1 range
    assert layer_parameters["init_forget_bias"] >= 0. and layer_parameters["init_forget_bias"] <= 1., "Forget bias for layer %s must be between 0-1" % layer_name
    assert layer_parameters["dropout_keep_prob"] >= 0. and layer_parameters["dropout_keep_prob"] <= 1., "dropout_keep_prob for layer %s must be between 0-1" % layer_name

    #verify that the output merge modes are either concat or sum or false if the layer is bidirectional, 
    #and that it is false if it is unidirectional
    #we also store the output sizes so that we can compare them to the input sizes in the next layer. notice that this requires
    # taking a look at the merge modes.
    #
    if layer_parameters['bidirectional']:

      if layer_parameters['output_merge_mode'] == False: #do nothing special to the bidirectional output, store it once for fw and once for bw
        output_sizes[layer_name] = [ layer_parameters['hidden_size'], layer_parameters['hidden_size'] ]
      elif layer_parameters['output_merge_mode'] == 'concat':
        output_sizes[layer_name] = [ layer_parameters['hidden_size'] * 2 ]
      elif layer_parameters['output_merge_mode'] == 'sum':
        output_sizes[layer_name] = [ layer_parameters['hidden_size'] ]
      else:
        raise ValueError("For a bidirectional layer, your merge most be one of the following list:\n%s" % permitted_bidirectional_output_merge_modes)
    else:
      if layer_parameters['output_merge_mode'] == False:
        output_sizes[layer_name] = [ layer_parameters['hidden_size'] ]
      else:
        raise ValueError("For a unidirectional layer, your merge most be one of the following list:\n%s" % permitted_unidirectional_output_merge_modes)

    #verify the dimensionality of the expected inputs at layer k equal the output dimensionalities from layers 0->k-1 that connect to k following k's merge mode
    if cur_layer_index == 0:
      assert layer_parameters['expected_input_size'] == -1, "The expected_input_size of the first layer in the stack must be -1. Instead it is %d" % layer_parameters['expected_input_size']
      assert len(layer_parameters['input_layers']) == 0, "Input layers for first layer in the stack must be an empty list"
    else:
      assert len(layer_parameters['input_layers']) > 0, "Input layers for all layers other than the first in the stack must be a list with >1 elements"
      
      #list off all the output sizes that will connect to this layer
      #print("Analyzing the inputs into layer %d, which expect an input size of %d" % (cur_layer_index, layer_parameters['expected_input_size']))
      connected_layer_sizes = []
      for l_name in layer_parameters['input_layers']:
        for sz in output_sizes[l_name]:
          connected_layer_sizes.append(sz) 
      #print("Connected layers to layer %d have sizes %s" % (cur_layer_index, str(connected_layer_sizes)))

      #verify the sum/concatenation of these layer sizes is equal to the expected input size
      if layer_parameters['input_merge_mode'] == 'sum':
        assert len(set(connected_layer_sizes)) == 1, "If using an elementwise summation of outputs from multiple layers, all layers need to output the same size. Instead, input sizes are %s" % (str(connected_layer_sizes))
        assert connected_layer_sizes[0] == layer_parameters['expected_input_size'], "Layer %s expected input size of %d differs from actual input size of %d" % (layer_name, layer_parameters['expected_input_size'], connected_layer_sizes[0])
      elif layer_parameters['input_merge_mode'] == 'concat':
        sum_connected_layers = sum(connected_layer_sizes)
        assert sum_connected_layers == layer_parameters['expected_input_size'], "Layer %s expected input size of %d differs from actual input size of %d" % (layer_name, layer_parameters['expected_input_size'], sum_connected_layers)

    cur_layer_index += 1
    last_layer = layer_name

  #now that we are all done, we should probably check that the final output layer isn't a list of outputs.
  #this is because this needs to be fed to the output projection.
  assert len(output_sizes.keys()) == cur_layer_index, "Expected an output size key for each layer processed. Have %d keys but final layer index reads %d" % (len(output_sizes.keys()), cur_layer_index) #sanity check, +1 because cur_layer_index start at 0
  return True

def _verify_decoder_state_initializer(stack_json, decoder_state_initializer):
  return True

def verify_encoder_decoder_stack_architecture(stack_json, decoder_state_initializer):

  print("Testing Encoder model architecture...")
  if _verify_recurrent_stack_architecture(stack_json['encoder']):
    print("Valid")
  else:
    print ("Invalid")
    return False

  print("Testing Decoder model state initialization...")
  if _verify_decoder_state_initializer(stack_json, decoder_state_initializer):
    print("Valid")
  else: 
    print ("Invalid")
    return False

  print("Testing Decoder model architecture...")
  if _verify_recurrent_stack_architecture(stack_json['decoder']):
    print("Valid")
  else:
    print("Invalid")
    return False

  return True

  

def load_encoder_decoder_architecture_from_json(json_file_path, decoder_state_initializer):
  with open(json_file_path, 'rb') as model_data:

    try:
      #notice we have an ordered dictionary. this is because we want to make sure we are going through the layers
      # in the right order on the verification, and when setting up the network itself. Python runs this is O(n) time
      # so this doesn't cost us really anything, and we get to have meaningful layer names.
      stack_model = json.load(model_data, object_pairs_hook=OrderedDict)
    except ValueError, e:
      print("Invalid json in %s" % json_file_path)
      print(e)
      raise
    
    print("Loaded JSON model architecture from %s" % json_file_path)
    print("This architecture will now be verified using decoder state initializer %s..." % decoder_state_initializer)

    if verify_encoder_decoder_stack_architecture(stack_model, decoder_state_initializer):
      return stack_model['encoder'], stack_model['decoder'] #this is the JSON object parsed as a python dict
    else:
      raise Exception, "Invalid architecture. Now dying"


#Probably delete this
def run_json_stack_architecture(model_json):
  pass


#TODO - eventually embeddings will be initialized with pre-trained FastText (on the dataset? full pretrained?)...seems like high overfitting potential, and sort of cheating...so make it a flag option.
def run_model(encoder_inputs,
              decoder_inputs,
              encoder_architecture,
              decoder_architecture,
              decoder_state_initializer,
              num_encoder_symbols,
              num_decoder_symbols,
              embedding_size,
              num_heads=1,
              output_projection=None,
              feed_previous=False,
              dtype=None,
              scope=None,
              initial_state_attention=False):

  #The first thing the model does is add an embedding to the encoder_inputs
  with variable_scope.variable_scope(scope or "model", dtype=dtype) as scope:
    dtype=scope.dtype

    #TODO - this is where the embedded decoder inputs should be extracted
    # why? because they might be pre-trained glove/word2vec/fasttext embeddings that aren't trained by the network
    #    BUT
    #      then they are still in memory for each bucket. can probably do better and abstract it into the argument
    #      to run_model()

    #First, we run the encoder on the inputs, which for now included the embedding transformation to the input 
    #before entering the rnn. this returns an encoder state which is an LSTMStateTuple
    #encoder_outputs, encoder_state = encoder.run_encoder(encoder_inputs,
    #                                        num_encoder_symbols,
    #                                        embedding_size,
    #                                        dtype=dtype)


    #encoder outputs are a list of length batch_size

    final_top_encoder_outputs, final_encoder_states = encoder.run_encoder_NEW(encoder_architecture,
                                                                              encoder_inputs,
                                                                              num_encoder_symbols,
                                                                              embedding_size,
                                                                              dtype=dtype)    

    #Then we create an attention state by reshaping the encoder outputs. This amounts to creating an additional
    #dimension, namely attention_length, so that our attention states are of shape [batch_size, atten_len=1, atten_size=size of last lstm output]
    #these attention states are used in every calculation of attention during the decoding process
    #we will use the STATE output from the decoder network as a query into the attention mechanism.
    attention_states = encoder.get_attention_state_from_encoder_outputs(final_top_encoder_outputs,
                                                                dtype=dtype)

    #then we run the decoder.
    return attention_decoder.embedding_attention_decoder(
          decoder_architecture,
          decoder_state_initializer,
          decoder_inputs,
          final_encoder_states, #this is a LIST of LSTMStateTuples or GRU States
          attention_states,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=None,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """

  #TODO - remove keyword argument here
  assert softmax_loss_function is not None, "Must have a defined loss function passed into the sequence loss by example"

  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.

        print("WARNING - NO DEFINED SOFTMAX LOSS FUNCTION")

        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(target, logit)






      """

      ROB, YOU LEFT OFF HERE.

      Find a way to get the target weight cross entropy gone for padded id words without explicitly storing it as a 
      trainable variable in the model file





      """
      #if weight.value==0.:
      #  assert(target==data_utils.PAD_ID, "If weight is 0, target id should be pad_id")
      #elif weight.value==1:
      #  assert(target.value!=data_utils.PAD_ID, "If weight is 1, target id should be anything other than the pad id")
      #else:
      #  print("weight came through that was not equal to 1.0 or 0.0")
      #  raise ValueError
     
      #print(weight.get_shape())
      log_perp_list.append(crossent * weight)







    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost







def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, tf.nn.rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputsut, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last"
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last"
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses
