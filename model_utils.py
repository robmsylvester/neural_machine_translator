from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import encoder
import attention_decoder
import json
from collections import OrderedDict

FLAGS = tf.app.flags.FLAGS

#TODO - redefine the GRU with my own implementation that adds support for default reset/update gates, activation function, initialization other than xavier
#TODO - add the initialization and activation parameters to the .json architecture
def _create_rnn_cell(json_layer_parameters, use_lstm=True):

  def _create_lstm(hidden_size, use_peepholes, init_forget_bias, dropout_keep_prob):
    c = core_rnn_cell_impl.LSTMCell(hidden_size, #number of units in the LSTM
                  use_peepholes=use_peepholes,
                  initializer=tf.contrib.layers.xavier_initializer(), #TODO - make this a json property.
                  state_is_tuple=True,
                  forget_bias=init_forget_bias)
    if dropout_keep_prob < 1.0:
      c = core_rnn_cell_impl.DropoutWrapper(c, output_keep_prob=dropout_keep_prob)
    return c

  #default hyperbolic tangent activation
  def _create_gru(hidden_size, dropout_keep_prob):
    g = core_rnn_cell_impl.GRUCell(hidden_size)
    if dropout_keep_prob < 1.0:
      g = core_rnn_cell_impl.DropoutWrapper(g, output_keep_prob=dropout_keep_prob)
    return g

  if use_lstm:
    return _create_lstm(json_layer_parameters['hidden_size'],
                        json_layer_parameters['peepholes'],
                        json_layer_parameters['init_forget_bias'],
                        json_layer_parameters['dropout_keep_prob'])
  else:
    return _create_gru(json_layer_parameters['hidden_size'],
                        json_layer_parameters['dropout_keep_prob'])




def _create_output_projection(target_size,
                              output_size):
  #Notice we don't put this in a scope. It lives in the sequence model itself and gets passed between layers of the
  #decoder. This is a pattern used without this file as these functions that begin with '_' are primarily helper functions.

  #We return a transpose variable itself to speed up training so we don't have to do this back and
  #forth between iterations. The weights transpose is necessary to pass to softmax. The weights themselves are used
  #in the attention decoder itself.
  weights_t = tf.get_variable("output_projection_weights", [target_size, output_size], dtype=tf.float32)
  weights = tf.transpose(weights_t)
  biases = tf.get_variable("output_projection_biases", [target_size], dtype=tf.float32)
  return (weights, biases, weights_t)


def _verify_recurrent_stack_architecture(stack_json, top_bidirectional_layer_allowed=False):

  #TODO - input validation on datatypes and input domains.
  """
  Verifies that the json inputs are valid architectures for an lstm or gru stack based on the interpreter in the encoder and decoder

  See the template file for a more detailed explanation

  Args:
  stack_json: Nested dictionary - Represents the encoder or decoder json.
              This is a result of calling the json module's load() function and passing the "encoder" or "decoder" key value.
  top_bidirectional_layer_allowed: Boolean - Whether or not the last layer in the stack can output a forward and a backward sequence (true), or just a forward sequence (false)

  Returns: Boolean - False if there are any violations in the arithmetic or parameter combinations, otherwise True

  """
  is_lstm = stack_json["use_lstm"]
  permitted_input_merge_modes = [False,'concat','sum'] #when multiple layers connect to an LSTM/GRU, what do we do with these inputs? concat or sum are the only options. first layer of stack must have value 'false'
  permitted_unidirectional_output_merge_modes = [False] #when a unidirectional LSTM/GRU has outputs for each time step, we dont have anything special to do. This is just for readability
  permitted_bidirectional_output_merge_modes = [False,'concat','sum'] #if all connected inputs will 'concat' or 'sum' their inputs, then you might as well merge the outputs right off the bat. Otherwise False

  cur_layer_index = 0
  output_sizes = {}
  
  #go one-by-one and make sure the architecture layer sizes add up
  for layer_name, layer_parameters in stack_json["layers"].iteritems():

    if cur_layer_index == 0 and layer_parameters["input_merge_mode"] != False:
        print("Input merge mode for first layer must be False")
        return False
    
    #merge mode is either concat or sum
    if layer_parameters["input_merge_mode"] not in permitted_input_merge_modes:
      print("Merge mode in %s is invalid" % layer_name)
      return False

    #no peephole connections on GRU's
    if not is_lstm and layer_parameters["peepholes"]:
        print("Cannot use peephole connections in layer %s because this is not an LSTM" % layer_name)
        return False

    #Forget bias and dropout probabilities are in 0-1 range
    if layer_parameters["init_forget_bias"] < 0. or layer_parameters["init_forget_bias"] > 1.:
      print("Forget bias for layer %s must be between 0-1" % layer_name)
      return False
    
    if layer_parameters["dropout_keep_prob"] < 0. or layer_parameters["dropout_keep_prob"] > 1.:
      print("dropout_keep_prob for layer %s must be between 0-1" % layer_name)
      return False

    #verify that the output merge modes are either concat or sum or false if the layer is bidirectional, 
    #and that it is false if it is unidirectional
    #we also store the output sizes so that we can compare them to the input sizes in the next layer. notice that this requires
    # taking a look at the merge modes.
    #
    if layer_parameters['bidirectional']:
      if layer_parameters['output_merge_mode'] == False: #do nothing special to the bidirectional output, store it once for fw and once for bw
        output_sizes[layer_name] = [ layer_parameters['hidden_size'], layer_parameters['hidden_size'] ] #this will be an lstm state tuple of two tensors, or it will be a gru output tensor
      elif layer_parameters['output_merge_mode'] == 'concat':
        output_sizes[layer_name] = [ layer_parameters['hidden_size'] * 2 ]
      elif layer_parameters['output_merge_mode'] == 'sum':
        output_sizes[layer_name] = [ layer_parameters['hidden_size'] ]
      else:
        print("For a bidirectional layer, your merge must be one of the following list:\n%s" % permitted_bidirectional_output_merge_modes)
        return False
    else:
      if layer_parameters['output_merge_mode'] == False:
        output_sizes[layer_name] = [ layer_parameters['hidden_size'] ]
      else:
        print("For a unidirectional layer, your merge must be one of the following list:\n%s" % permitted_unidirectional_output_merge_modes)
        return False

    #verify the dimensionality of the expected inputs at layer k equal the output dimensionalities from layers 0->k-1 that connect to k following k's merge mode
    if cur_layer_index == 0:
      if not layer_parameters['expected_input_size'] == -1:
        print("The expected_input_size of the first layer in the stack must be -1. Instead it is %d" % layer_parameters['expected_input_size'])
        return False
      if not len(layer_parameters['input_layers']) == 0:
        print("Input layers for first layer in the stack must be an empty list")
        return False
    else:
      if not len(layer_parameters['input_layers']) > 0:
        print("Input layers for all layers other than the first in the stack must be a list with >1 elements")
        return False
      
      #list off all the output sizes that will connect to this layer
      #print("Analyzing the inputs into layer %d, which expect an input size of %d" % (cur_layer_index, layer_parameters['expected_input_size']))
      connected_layer_sizes = []
      for l_name in layer_parameters['input_layers']:
        if l_name not in output_sizes.keys():
          print("Layer name %s not found in list of output layer names. Check .json keys" % l_name)
          return False
        for sz in output_sizes[l_name]:
          connected_layer_sizes.append(sz) 
      #print("Connected layers to layer %d have sizes %s" % (cur_layer_index, str(connected_layer_sizes)))

      #verify the sum/concatenation of these layer sizes is equal to the expected input size
      if layer_parameters['input_merge_mode'] == 'sum':
        if not len(set(connected_layer_sizes)) == 1:
          print("If using an elementwise summation of outputs from multiple layers, all layers need to output the same size. Instead, input sizes are %s" % (str(connected_layer_sizes)))
          return False
        if not connected_layer_sizes[0] == layer_parameters['expected_input_size']:
          print("Layer %s expected input size of %d differs from actual input size of %d" % (layer_name, layer_parameters['expected_input_size'], connected_layer_sizes[0]))
          return False
      elif layer_parameters['input_merge_mode'] == 'concat':
        sum_connected_layers = sum(connected_layer_sizes)
        if not sum_connected_layers == layer_parameters['expected_input_size']:
          print("Layer %s expected input size of %d differs from actual input size of %d" % (layer_name, layer_parameters['expected_input_size'], sum_connected_layers))
          return False
    cur_layer_index += 1
    last_layer = layer_name

  #now that we are all done, we should probably check that the final output layer isn't a list of outputs.
  #this is because this needs to be fed to the output projection.
  if not len(output_sizes.keys()) == cur_layer_index:
    print("If this message is seen there is a bug. Expected an output size key for each layer processed. Have %d keys but final layer index reads %d" % (len(output_sizes.keys()), cur_layer_index)) #sanity check, +1 because cur_layer_index start at 0
    return False
  if stack_json["layers"][last_layer]["bidirectional"] and not top_bidirectional_layer_allowed:
    print("The top layer cannot be bidirectional as per the default settings passed in verify_encoder_decoder_stack_architecture")
    return False
  
  #we made it! party on wayne!
  return True

def _verify_decoder_state_initializer(stack_json, decoder_state_initializer):
  if decoder_state_initializer=="nematus":
    print("Nematus decoder state initializer is not yet supported")
    return False
  elif decoder_state_initializer=="top_layer_mirror":
    last_layer_name = next(reversed(stack_json['encoder']['layers']))
    last_layer_parameters = stack_json['encoder']['layers'][last_layer_name]
    
    state_size = last_layer_parameters['hidden_size']
    state_bidirectional = last_layer_parameters['bidirectional']
    
    for l_name, l_params in stack_json['decoder']['layers'].iteritems():
      if l_params['bidirectional'] != state_bidirectional:
        print("If using top layer mirror, every decoder layer must have the same value for the bidirectional property as the top layer in the encoder. Layer %s reads %s, top encoder layer reads %s" %(l_name, str(l_params['bidirectional']), str(state_bidirectional)))
        return False
      elif state_size != l_params['hidden_size']:
        print("If using top layer mirror, every decoder layer must have the same value for the hidden size property as the top layer in the encoder. Layer %s reads %d, top encoder layer reads %d" %(l_name, l_params['hidden_size'], state_size))
        return False

  return True


#TODO - not much to do here other than verify dimensionality expansion with the variable number of attention heads is working properly
def _verify_attention_architecture(stack_json):
  return True

def verify_encoder_decoder_stack_architecture(stack_json, decoder_state_initializer):

  print("Testing Encoder model architecture...")
  if not _verify_recurrent_stack_architecture(stack_json['encoder'], top_bidirectional_layer_allowed=True):
    print ("Invalid Encoder Architecture.")
    return False

  print("Testing Attention mechanism architecture...")
  if not _verify_attention_architecture(stack_json):
    print("Invalid Attention Architecture")
    return False

  print("Testing Decoder model state initialization...")
  if not _verify_decoder_state_initializer(stack_json, decoder_state_initializer):
    print ("Invalid")
    return False

  print("Testing Decoder model architecture...")
  if not _verify_recurrent_stack_architecture(stack_json['decoder'], top_bidirectional_layer_allowed=False):
    print("Invalid")
    return False

  return True

  
def load_encoder_decoder_architecture_from_json(json_file_path, decoder_state_initializer):
  with open(json_file_path, 'rb') as model_data:

    try:
      #notice we have an ordered dictionary. this is because we want to make sure we are going through the layers
      # in the right order on the verification, and when setting up the network itself. Python runs O(n) ordered dict ops
      # so this doesn't cost us a lot, and we get to have meaningful layer names.
      stack_model = json.load(model_data, object_pairs_hook=OrderedDict)
    except ValueError, e:
      print("Invalid json in %s" % json_file_path)
      print(e)
      raise
    
    print("Loaded JSON model architecture from %s" % json_file_path)

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

    #TODO - this is probably where the embedded decoder inputs should be extracted

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

    #print("Attention Mechanism Shape is %s" % attention_states.get_shape())

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



#TODO - rewrite the args to this
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
