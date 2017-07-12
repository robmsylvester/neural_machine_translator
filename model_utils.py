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



def _verify_recurrent_stack_architecture(stack_json, top_bidirectional_layer_allowed=False):
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
  if not top_bidirectional_layer_allowed and stack_json["layers"][last_layer]["bidirectional"] == True:
    print("The top layer cannot be bidirectional. This is the default in the decoder until more support is added for dealing with the final output when the merge modes are not defined")
    return False
  return True

def _verify_decoder_state_initializer(stack_json, decoder_state_initializer):
  return True

def verify_encoder_decoder_stack_architecture(stack_json, decoder_state_initializer):

  #TODO - the top bidirectional layer allowed argument can probably go away. but test it better before
  # removing it here and in the actual method.
  print("Testing Encoder model architecture...")
  if _verify_recurrent_stack_architecture(stack_json['encoder'], top_bidirectional_layer_allowed=True):
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
  if _verify_recurrent_stack_architecture(stack_json['decoder'], top_bidirectional_layer_allowed=False):
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



def _get_residual_layer_inputs_as_list(current_layer_name, current_layer_input_list, all_previous_layer_outputs):
  #Examines a current layer list's input layers, and looks at all the previous layer outputs and puts them in a list
  # if they are outputs from any of those listed layers
  #Args:
  # current_layer_name - string, the name of the current layer for which we are building a list
  # current_layer_input_list - list of strings, with each string being one of the possible previous layer names. this represents the names of layers that connect to this layer
  # all_previous_layer_outputs - list of lists of tensors, with each of these inner lists containing output tensors. this inner list will have a length of one or two most likely (unidirectional or bidirectional)
  #Returns:
  # list of previous layer outputs to use as inputs to this layer. can be an empty list (as well it always will be for the first layer)
  input_list = []

  #add to inputs the residual connections that connect to this layer
  for candidate_layer_name, candidate_layer_output in all_previous_layer_outputs.iteritems():
    #print("layer %s is checking layer %s for inputs" %  (layer_name, candidate_layer_name))
    if candidate_layer_name == current_layer_name: #layers don't have residual connections to themselves...
      #print("skipping")
      continue
    elif candidate_layer_name in current_layer_input_list: #we will gather the layers that are marked in the architecture
      #print("layer %s will use input from layer %s" % (layer_name, candidate_layer_name))
      for layer_output in candidate_layer_output: #candidate_layer_output is always a list, sometimes with 1 element, sometimes with 2
        input_list.append(layer_output)

  return input_list



def _combine_residual_inputs(inputs_as_list, merge_mode, return_list=True):
  if merge_mode == 'concat':
    inputs=tf.concat(inputs_as_list, axis=2) #creates tensor of shape (max_time, batch_size, input_size)
  elif merge_mode == 'sum':
    inputs = tf.add_n(inputs_as_list)
  else:
    raise ValueError("Input merge mode must be either concat or sum")

  #unstacking the inputs will create a list of tensors with length max_time, where each tensor has shape (batch_size, input_size)
  return tf.unstack(inputs) if return_list else inputs



def encoder_decoder_attention(encoder_inputs,
                                decoder_inputs,
                                encoder_input_lengths,
                                decoder_input_lengths,
                                encoder_architecture,
                                decoder_architecture,
                                decoder_state_initializer,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                embedding_algorithm="network", #network means random initialization, train via backprop. otherwise use fastext, word2vec or glove
                                train_embeddings=True, #if true and not network for embedding algorithm, will use unsupervised algorithm for initialization and train via backprop, which dominates anyway
                                num_heads=1, #attention heads to read. 
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

    #encoder outputs are a list of length max_encoder_output of tensors with the shape of the top layer
    if FLAGS.encoder_rnn_api == "dynamic":
      final_top_encoder_outputs, final_encoder_states = encoder.dynamic_embedding_encoder(encoder_architecture,
                                                                                      encoder_inputs,
                                                                                      encoder_input_lengths,
                                                                                      num_encoder_symbols,
                                                                                      embedding_size,
                                                                                      embedding_algorithm=embedding_algorithm,
                                                                                      train_embeddings=train_embeddings,
                                                                                      dtype=dtype)
    elif FLAGS.encoder_rnn_api == "static":
      final_top_encoder_outputs, final_encoder_states = encoder.static_embedding_encoder(encoder_architecture,
                                                                                          encoder_inputs,
                                                                                          encoder_input_lengths,
                                                                                          num_encoder_symbols,
                                                                                          embedding_size,
                                                                                          embedding_algorithm=embedding_algorithm,
                                                                                          train_embeddings=train_embeddings,
                                                                                          dtype=dtype)
    else:
      raise ValueError, "encoder rnn api must be dynamic or static. eventually move this to flags check function"

    #Then we create an attention state by reshaping the encoder outputs. This amounts to creating an additional
    #dimension, namely attention_length, so that our attention states are of shape [batch_size, atten_len=1, attention_size=size of last lstm output]
    #these attention states are used in every calculation of attention during the decoding process
    #we will use the STATE output from the decoder network as a query into the attention mechanism.
    attention_states = encoder.reshape_encoder_outputs_for_attention(final_top_encoder_outputs,
                                                                    dtype=dtype)

    print(attention_states.get_shape())

    #then we run the decoder.
    return attention_decoder.embedding_attention_decoder(
          decoder_architecture,
          decoder_state_initializer,
          decoder_inputs,
          decoder_input_lengths,
          final_encoder_states, #this is a list of lists of LSTMStateTuples or GRU States
          attention_states,
          num_decoder_symbols,
          embedding_size,
          embedding_algorithm=embedding_algorithm,
          train_embeddings=train_embeddings,
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



      #TODO - INSERT BOOSTING on average sentence score per logit per some number of iterations


      #Sanity Check
      #if weight==0.:
      #  assert(target == data_utils.PAD_ID, "If weight is 0, target id should be pad_id")
      #elif weight==1:
      #  assert(target != data_utils.PAD_ID, "If weight is 1, target id should be anything other than the pad id")
      #else:
      #  print("weight came through that was not equal to 1.0 or 0.0")
      #  raise ValueError

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




def run_eda_architecture(encoder_inputs,
                       decoder_inputs,
                       max_encoder_length,
                       max_decoder_length,
                       encoder_input_lengths,
                       decoder_input_lengths,
                       targets,
                       weights,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    max_encoder_length: An int, truncate encoder inputs past this number
    max_decoder_length: An int, truncate decoder inputs past this number
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    seq2seq: A sequence-to-sequence model function; it takes 4 input that
      agree with encoder_inputs and decoder_inputs, and encoder_input_lengths
      and decoder_input_lengths.
      returns consist of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_without_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.
  """
  all_inputs = encoder_inputs + decoder_inputs + targets + weights

  with ops.name_scope(name, "model_without_buckets", all_inputs):
    #with variable_scope.variable_scope(
    #    variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
    with variable_scope.variable_scope(variable_scope.get_variable_scope()):

      assert len(encoder_inputs) == max_encoder_length
      assert len(decoder_inputs) == max_decoder_length+1

      outputs, _ = seq2seq(encoder_inputs[:max_encoder_length],
                           decoder_inputs[:max_decoder_length], #TODO check not off by one here
                           encoder_input_lengths,
                           decoder_input_lengths)

      if per_example_loss:
        losses = sequence_loss_by_example(
                  outputs,
                  targets[:max_decoder_length],
                  weights[:max_decoder_length],
                  softmax_loss_function=softmax_loss_function)
      else:

        losses = sequence_loss(
                outputs,
                targets[:max_decoder_length],
                weights[:max_decoder_length],
                softmax_loss_function=softmax_loss_function)

  return outputs, losses
