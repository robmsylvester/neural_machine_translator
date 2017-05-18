import tensorflow as tf
import custom_core_rnn as core_rnn
from tensorflow.python import shape
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

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


#TODO - this needs to become dynamic instead of static rnn's, probably. depending on timer calls. fuckin' padding.
def run_encoder_NEW(encoder_json,
                    encoder_inputs,
                    num_encoder_symbols,
                    embedding_size,
                    dtype=None):


  # embeddings need to become non-updated-by-backprop embeddings from unsupervised glove, word2vec, or fasttext options
  with variable_scope.variable_scope("encoder", dtype=dtype) as scope:
    dtype=scope.dtype

    #create embeddings - this will eventually be moved elsewhere when the embeddings are no longer trained
    with variable_scope.variable_scope("embeddings") as scope:
      embeddings = tf.get_variable("encoder_embeddings",
                                  shape=[num_encoder_symbols, embedding_size],
                                  initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                  dtype=dtype)

      #get the embedded inputs from the lookup table
      embedded_encoder_inputs = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    current_layer = 0
    cell_outputs_states = {} #indexed by layer name in json, which stores a dict with keys "output" and "state"

    #TODO - this NEEDS to become dynamic rnn
    for layer_name, layer_parameters in encoder_json["layers"].iteritems():

      #cell outputs and states are created. if unidirectional gru or lstm, out is still a list.
      #if bidirectional, it is a list of two tensors, unless they are concatenated
      cell_outputs[layer_name] = []

      #TODO, take this and the loop and refactor into a get_inputs function
      #first loop will get embeddings, later loops will use previous iteration outputs, plus any residual connections
      input_list = []

      #add to inputs the residual connections that connect to this layer
      for candidate_layer_name, candidate_layer_output in cell_outputs.iteritems():
        if candidate_layer_name == layer_name: #layers don't have residual connections to themselves...
          continue
        elif candidate_layer_name in layer_parameters['input_layers'] #we will gather the layers that are marked in the architecture
          print("Layer %s will use input from layer %s" % (layer_name, candidate_layer_name))
          input_list.append(candidate_layer_output)

      if len(input_list) == 0:
        print("layer %s has no inputs in the input list. it will use the embeddings" % layer_name)
        inputs = embedded_encoder_inputs

      #combine these residual inputs following this layer's merge mode
      if layer_parameters['input_merge_mode'] == 'concat':
      elif layer_parameters['input_merge_mode'] == 'sum':
      else:
        raise ValueError("Only concat and sum are built for input merge modes in the runtime encoder stack loop, so far...")

      #create/get the cells, and run them
      #this will give us the outputs, and we can combine them as necessary
      #c stands for cell, out for outputs, f for forward, b for backward
      if layer_parameters['bidirectional']:
        cf = _create_encoder_cell(layer_parameters)
        cb = _create_encoder_cell(layer_parameters)

        out_fb, out_f, out_b, state_f, state_b = core_rnn.static_bidirectional_rnn(cf, cb, inputs, dtype=dtype)
        print("shape of out_forward for layer %s is %s " % (layer_name, str(out_f.get_shape())))

        #store the outputs according to how they have to be merged.
        #they will be a list with 2 elements, the forward and backward outputs. or, a list with one element, the concatenation or sum of the 2 elements.
        if layer_parameters['output_merge_mode'] == 'concat':
          assert out_f.get_shape().ndims == 2
          cell_outputs_states[layer_name].append(tf.concat([out_f, out_b], axis=1)) #TODO - need to unstack here?
        elif layer_parameters['output_merge_mode'] == 'sum':
          cell_outputs_states[layer_name].append(tf.unstack(tf.add_n([out_f, out_b]))) #TODO - need to unstack here?
        else:
          cell_outputs_states[layer_name].append(out_f)
          cell_outputs_states[layer_name].append(out_b)

      else:
        cf = _create_encoder_cell(layer_parameters)
        out_f, state_f = core_rnn.static_rnn(cf, inputs, dtype=dtype)
        cell_outputs_states[layer_name]["out"].append(out_f)

      #we do care about the states, but only the top most state, which is what is stored in state_f




    #with variable_scope(layer_name, dtype=dtype) as scope:
    #   create_encoder_cell(layer_parameters)

    #current_layer += 1
    pass


#TODO - modularize this from javascript object and replace with the functions above
def run_encoder(encoder_inputs,
            num_encoder_symbols,
            embedding_size,
            dtype=None):

  with variable_scope.variable_scope("encoder", dtype=dtype) as scope:
    dtype = scope.dtype
    #print("at the beginning of the encoder_rnn layer 1, the scope name is %s" % scope.name)
    #First layer is a bidirectional lstm with embedding wrappers
    with variable_scope.variable_scope("encoder_fw1") as scope:

      fw1 = _create_encoder_lstm(FLAGS.encoder_hidden_size,
                          FLAGS.encoder_use_peepholes,
                          FLAGS.encoder_init_forget_bias,
                          FLAGS.encoder_dropout_keep_probability)
      encoder_fw1 = core_rnn_cell.EmbeddingWrapper(fw1, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
    
    with variable_scope.variable_scope("encoder_bw1") as scope:
      bw1 = _create_encoder_lstm(FLAGS.encoder_hidden_size,
                          FLAGS.encoder_use_peepholes,
                          FLAGS.encoder_init_forget_bias,
                          FLAGS.encoder_dropout_keep_probability)
      encoder_bw1 = core_rnn_cell.EmbeddingWrapper(bw1, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
    
    with variable_scope.variable_scope("encoder_bidir1") as scope:
      #NOTE - these outputs have 2x the side of the the forward and backward lstm outputs.
      outputs1, outputs1_fw, outputs1_bw, _, _ = core_rnn.static_bidirectional_rnn(encoder_fw1, encoder_bw1, encoder_inputs, dtype=dtype)

    #The second layer is a also bidirectional
    with variable_scope.variable_scope("encoder_fw2") as scope:
      fw2 = _create_encoder_lstm(FLAGS.encoder_hidden_size,
                          FLAGS.encoder_use_peepholes,
                          FLAGS.encoder_init_forget_bias,
                          FLAGS.encoder_dropout_keep_probability)

    with variable_scope.variable_scope("encoder_bw2") as scope:
      bw2 = _create_encoder_lstm(FLAGS.encoder_hidden_size,
                          FLAGS.encoder_use_peepholes,
                          FLAGS.encoder_init_forget_bias,
                          FLAGS.encoder_dropout_keep_probability)

    with variable_scope.variable_scope("encoder_bidir2") as scope:
      #NOTE - these outputs have 2x the size of the the forward and backward lstm outputs.
      outputs2, outputs2_fw, outputs2_bw, _, _ = core_rnn.static_bidirectional_rnn(fw2, bw2, outputs1, dtype=dtype)


    #The third layer is a unidirectional lstm with a residual connection to the first layer outputs
    #This means we need to add the outputs element-wise, but then unstack them as a list once again
    #because a STATIC rnn asks for a list input of tensors, not just a tensor with one more
    #dimension as would be given by tf.add_n
    with variable_scope.variable_scope("encoder_fw3") as scope:
      
      inputs3 = tf.unstack(tf.add_n([outputs1,outputs2], name="residual_encoder_layer3_input"))
      fw3 = _create_encoder_lstm(FLAGS.encoder_hidden_size,
                          FLAGS.encoder_use_peepholes,
                          FLAGS.encoder_init_forget_bias,
                          FLAGS.encoder_dropout_keep_probability)
      outputs3, _ = core_rnn.static_rnn(fw3, inputs3, dtype=dtype)


    #The fourth layer is a unidirectional lstm with a residual connection to the second layer outptus
    #However, the dimensions do not quite add up if we pass the raw bidirecitonal output, so we will
    #instead pass a summation of the bidirectional forward, the bidirectional backward, and the third layer
    with variable_scope.variable_scope("encoder_fw4") as scope:
      
      inputs4 = tf.unstack(tf.add_n([outputs2_fw, outputs2_bw, outputs3], name="residual_encoder_layer4_input"))
      fw4 = _create_encoder_lstm(FLAGS.encoder_hidden_size,
                          FLAGS.encoder_use_peepholes,
                          FLAGS.encoder_init_forget_bias,
                          FLAGS.encoder_dropout_keep_probability)
      encoder_outputs, encoder_state = core_rnn.static_rnn(fw4, inputs4, dtype=dtype)
      #print("at the end of the encoder_rnn layer 4, the scope name is %s" % scope.name)
    
    #Make sure everything went alright
    #TODO - turn these OFF when not testing to speed things up.
    assert type(encoder_state) is core_rnn_cell_impl.LSTMStateTuple, "encoder_state should be an LSTMStateTuple. state is tuple is now true by default in TF."
    assert len(encoder_outputs) == len(encoder_inputs), "encoder input length (%d) should equal encoder output length(%d)" % (len(encoder_inputs), len(encoder_outputs))
    return encoder_outputs, encoder_state

# Concatenation of encoder outputs to put attention on.
# TODO - encoder hidden size here is written here instead of output_size
# way because it is assumed top layer is not bidirectional, or otherwise
# have any reason to use a different variable.
def get_attention_state_from_encoder_outputs(encoder_outputs,
                                            scope=None,
                                            dtype=None):
  with variable_scope.variable_scope(scope or "attention_from_encoder_outputs", dtype=dtype) as scope:
    dtype=scope.dtype
    top_states = [array_ops.reshape(enc_out, [-1, 1, FLAGS.encoder_hidden_size]) for enc_out in encoder_outputs]
    attention_states = array_ops.concat(top_states, 1)
    return attention_states 