import tensorflow as tf
import custom_core_rnn as core_rnn
import embeddings
import model_utils
from tensorflow.python import shape
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from collections import OrderedDict
FLAGS = tf.app.flags.FLAGS

def _create_encoder_cell(json_layer_parameters):
  if json_layer_parameters['lstm']:
    return _create_encoder_lstm(json_layer_parameters['hidden_size'],
                                json_layer_parameters['peepholes'],
                                json_layer_parameters['init_forget_bias'],
                                json_layer_parameters['dropout_keep_prob'])
  else:
    return _create_encoder_gru(json_layer_parameters['hidden_size'],
                                json_layer_parameters['peepholes'],
                                json_layer_parameters['init_forget_bias'],
                                json_layer_parameters['dropout_keep_prob'])

def _create_encoder_lstm(hidden_size, use_peepholes, init_forget_bias, dropout_keep_prob):
  c = core_rnn_cell_impl.LSTMCell(hidden_size, #number of units in the LSTM
                use_peepholes=use_peepholes,
                initializer=tf.contrib.layers.xavier_initializer(),
                forget_bias=init_forget_bias)
  if dropout_keep_prob < 1.0:
    c = core_rnn_cell_impl.DropoutWrapper(c, output_keep_prob=dropout_keep_prob)
  return c

def _create_encoder_gru(hidden_size, init_forget_bias, dropout_keep_prob):
  raise NotImplementedError

# Concatenation of encoder outputs to put attention on.
# have any reason to use a different variable.
def reshape_encoder_outputs_for_attention(encoder_outputs,
                                          scope=None,
                                          dtype=None):
  #Reshapes your list of encoder outputs at each time step into the dimensionality of the attention mechanism
  #Args
  # encoder_outputs - list of length max_time with encoder outputs of shape (batch_size, hidden_size), where hidden_size may be 2x hidden size if you had a bidirectional top encoder layer
  # scope= tensorflow scope with which to launch this code
  # dtype= tensorflow datatype for this weight parameter

  #Returns
  # A single tensor of

  with variable_scope.variable_scope(scope or "attention_from_encoder_outputs", dtype=dtype) as scope:
    assert isinstance(encoder_outputs, (list)), "Encoder outputs must be a python list. Instead it is of type %s" % str(type(encoder_outputs))
    #print("Length of encoder outputs is %d" % len(encoder_outputs))
    dtype=scope.dtype
    #print("get attention state from encoder outputs has been called.")
    output_size = encoder_outputs[0].get_shape().with_rank(2)[1].value

    top_states = [array_ops.reshape(enc_out, [-1, 1, output_size]) for enc_out in encoder_outputs]
    attention_states = array_ops.concat(top_states, 1) #does this need to be axis=2
    return attention_states 


def dynamic_embedding_encoder(encoder_json,
                              encoder_inputs,
                              encoder_input_lengths,
                              num_encoder_symbols,
                              embedding_size,
                              embedding_algorithm="network",
                              train_embeddings=True,
                              nematus_state_values=False, #tracks every single state value at every time step
                              dtype=None):

  with variable_scope.variable_scope("dynamic_encoder", dtype=dtype) as scope:
    dtype = scope.dtype

    embedded_encoder_inputs, _ = embeddings.get_word_embeddings(encoder_inputs,
                                                            num_encoder_symbols,
                                                            embedding_size,
                                                            embed_language="source",
                                                            scope_name="encoder_embeddings",
                                                            embed_algorithm=embedding_algorithm,
                                                            train_embeddings=train_embeddings,
                                                            return_list=False,
                                                            dtype=dtype)

    cell_outputs = OrderedDict() #indexed by layer name in json
    cell_states = OrderedDict()

    for layer_name, layer_parameters in encoder_json["layers"].iteritems(): #this is an ordered dict
      with variable_scope.variable_scope(layer_name) as scope:

        #first loop will get embeddings, later loops will use previous iteration outputs, plus any residual connections
        input_list = model_utils._get_residual_layer_inputs_as_list(layer_name, layer_parameters['input_layers'], cell_outputs)

        inputs = model_utils._combine_residual_inputs(input_list, layer_parameters['input_merge_mode'], return_list=False) if len(input_list) else embedded_encoder_inputs

        #create/get the cells, and run them
        #this will give us the outputs, and we can combine them as necessary
        #c stands for cell, out for outputs, f for forward, b for backward
        if layer_parameters['bidirectional']:
          cf = _create_encoder_cell(layer_parameters)
          cb = _create_encoder_cell(layer_parameters)

          out_fb, state_fb = tf.nn.bidirectional_dynamic_rnn(cf,
                                                            cb,
                                                            inputs,
                                                            sequence_length=encoder_input_lengths,
                                                            dtype=dtype,
                                                            time_major=True) #Remember, this is because we use (max_time, batch_size, input size)
          
          out_f, out_b = out_fb #unpack the output tuples to forward and backward
          state_f, state_b = state_fb #unpack the final states to forward and backward

          #store the outputs according to how they have to be merged.
          #they will be a list with 2 elements, the forward and backward outputs. or, a list with one element, the concatenation or sum of the 2 elements.
          
          if layer_parameters['output_merge_mode'] == 'concat':
            #assert out_f[0].get_shape().ndims == 2
            cell_outputs[layer_name] = [tf.concat(out_fb, 2)]
          elif layer_parameters['output_merge_mode'] == 'sum':
            #assert out_f[0].get_shape().ndims == 2
            #cell_outputs[layer_name] = [tf.unstack(tf.add_n([out_f, out_b]))] #unstack recreates a list of length bucket_length of tensors with shape (batch_size, hidden_size)
            cell_outputs[layer_name] = [tf.add_n([out_f, out_b])]
          else:
            cell_outputs[layer_name] = [out_f, out_b]

          #out_f is a list, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          cell_states[layer_name] = [state_f,state_b]

        else:
          cf = _create_encoder_cell(layer_parameters)

          #out_f is a list of tensor outputs, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          out_f, state_f = tf.nn.dynamic_rnn(cf,
                                            inputs,
                                            sequence_length=encoder_input_lengths,
                                            dtype=dtype,
                                            time_major=True)

          #print("unidirectional rnn ran.\n\ttype of out_f is %s\n\ttype of state_f is %s" % (str(type(out_f)),str(type(state_f))))
          cell_outputs[layer_name] = [out_f]
          cell_states[layer_name] = [state_f]

      top_layer = layer_name # just for readability

    if FLAGS.decoder_state_initializer == 'nematus':
      raise NotImplementedError, "Haven't written the nematus decoder state initializer yet, and currently this function only returns and tracks top-level states in the loop."

    #We do this anyway outside of the if statement, because the concatenation won't do anything otherwise, and the unstack 
    #If they had previously merged them via an explicit call to concat or sum, that this would be fine.
    if len(cell_outputs[top_layer]) > 1:
      print("WARNING - your top layer cell outputs more than a single tensor at each time step. Perhaps it is bidirectional with no output merge mode specified. These tensors will be concatenated along their final axis. You should change this in the JSON to be 'concat' for readability, or 'sum' if you want the tensors element-wise added.")
    
    stack_output = tf.unstack(tf.concat(cell_outputs[top_layer], axis=2))

    stack_states = cell_states.values()

    #This output should be a list with one element
    return stack_output, stack_states



def static_embedding_encoder(encoder_json,
                          encoder_inputs,
                          encoder_input_lengths,
                          num_encoder_symbols,
                          embedding_size,
                          embedding_algorithm=None,
                          train_embeddings=True,
                          nematus_state_values=False, #TODO.....but really not that important right now
                          dtype=None):


  # embeddings need to become non-updated-by-backprop embeddings from unsupervised glove, word2vec, or fasttext options
  with variable_scope.variable_scope("static_encoder", dtype=dtype) as scope:
    dtype=scope.dtype

    embedded_encoder_inputs, _ = embeddings.get_word_embeddings(encoder_inputs,
                                                            num_encoder_symbols,
                                                            embedding_size,
                                                            embed_language="source",
                                                            scope_name="encoder_embeddings",
                                                            embed_algorithm=embedding_algorithm,
                                                            train_embeddings=train_embeddings,
                                                            return_list=True,
                                                            dtype=dtype)

    cell_outputs = OrderedDict() #indexed by layer name in json
    cell_states = OrderedDict()

    for layer_name, layer_parameters in encoder_json["layers"].iteritems(): #this is an ordered dict

      with variable_scope.variable_scope(layer_name) as scope:
        
        #first loop will get embeddings, later loops will use previous iteration outputs, plus any residual connections
        input_list = model_utils._get_residual_layer_inputs_as_list(layer_name, layer_parameters['input_layers'], cell_outputs)
        #print("layer %s has %d total inputs in the input list." % (layer_name, len(input_list)))

        inputs = model_utils._combine_residual_inputs(input_list, layer_parameters['input_merge_mode'], return_list=True) if len(input_list) else embedded_encoder_inputs

        #create/get the cells, and run them
        #this will give us the outputs, and we can combine them as necessary
        #c stands for cell, out for outputs, f for forward, b for backward
        if layer_parameters['bidirectional']:
          cf = _create_encoder_cell(layer_parameters)
          cb = _create_encoder_cell(layer_parameters)
          out_f, out_b, state_f, state_b = core_rnn.static_bidirectional_rnn(cf, cb, inputs, dtype=dtype)

          #store the outputs according to how they have to be merged.
          #they will be a list with 2 elements, the forward and backward outputs. or, a list with one element, the concatenation or sum of the 2 elements.
          
          if layer_parameters['output_merge_mode'] == 'concat':
            # Concat each of the forward/backward outputs
            flat_out_f = nest.flatten(out_f)
            flat_out_b = nest.flatten(out_b)

            flat_outputs = tuple(
                array_ops.concat([fw, bw], 1)
                for fw, bw in zip(flat_out_f, flat_out_b))

            out_fb = nest.pack_sequence_as(structure=out_f,
                                          flat_sequence=flat_outputs)

            cell_outputs[layer_name] = [out_fb]
          elif layer_parameters['output_merge_mode'] == 'sum':
            cell_outputs[layer_name] = [tf.unstack(tf.add_n([out_f, out_b]))] #unstack recreates a list of length bucket_length of tensors with shape (batch_size, hidden_size)
          else:
            cell_outputs[layer_name] = [out_f, out_b]

          #out_f is a list, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          cell_states[layer_name] = [state_f,state_b]

        else:
          cf = _create_encoder_cell(layer_parameters)

          #out_f is a list of tensor outputs, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          out_f, state_f = core_rnn.static_rnn(cf, inputs, dtype=dtype)
          #print("unidirectional rnn ran.\n\ttype of out_f is %s\n\ttype of state_f is %s" % (str(type(out_f)),str(type(state_f))))
          cell_outputs[layer_name] = [out_f]
          cell_states[layer_name] = [state_f]

      top_layer = layer_name # just for readability

    #end main loop
    #we do care about the states, but only the top most state, which is what is stored in state_f unless it is bidirectional
    #we do care about the outputs, but they might be bidirectional outputs too, so we must check that as well
    #for now, we will just concatenate these by default

    if FLAGS.decoder_state_initializer == 'nematus':
      raise NotImplementedError, "Haven't written the nematus decoder state initializer yet, and currently this function only returns and tracks top-level states in the loop."

    #We do this anyway outside of the if statement, because the concatenation won't do anything otherwise, and the unstack 
    if len(cell_outputs[top_layer]) > 1:
      print("WARNING - your top layer cell outputs more than a single tensor at each time step. Perhaps it is bidirectional with no output merge mode specified. These tensors will be concatenated along their final axis. You should change this in the JSON to be 'concat' for readability, or 'sum' if you want the tensors element-wise added.")
    
    stack_output = tf.unstack(tf.concat(cell_outputs[top_layer], axis=2)) #we want a list
    stack_states = cell_states.values()

    #This output should be a list with one element
    return stack_output, stack_states