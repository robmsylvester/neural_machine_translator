import tensorflow as tf
import custom_core_rnn as core_rnn
from tensorflow.python import shape
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
import json
from collections import OrderedDict


FLAGS = tf.app.flags.FLAGS

def _create_encoder_lstm(hidden_size, use_peepholes, init_forget_bias, dropout_keep_prob):
  c = core_rnn_cell_impl.LSTMCell(hidden_size, #number of units in the LSTM
                use_peepholes=use_peepholes,
                initializer=tf.contrib.layers.xavier_initializer(),
                forget_bias=init_forget_bias)
  if dropout_keep_prob < 1.0:
    c = core_rnn_cell_impl.DropoutWrapper(c, output_keep_prob=dropout_keep_prob)
  return c

def verify_encoder_architecture(encoder_json):
  permitted_merge_modes = ['concat', 'sum']

  cur_layer = 0
  output_sizes = {}
  
  #go one-by-one and make sure the architecture adds up
  for layer_name, layer_parameters in encoder_json["layers"].iteritems():
    
    #merge mode is either concat or sum
    assert layer_parameters["merge_mode"] in permitted_merge_modes, "Merge mode in %s is invalid" % layer_name

    #no peephole connections on GRU's
    if not layer_parameters["lstm"]:
      assert not layer_parameters["peepholes"], "Cannot use peephole connections in layer %s because this is not an LSTM" % layer_name

    #Forget bias and dropout probabilities are in 0-1 range
    assert layer_parameters["init_forget_bias"] >= 0. and layer_parameters["init_forget_bias"] <= 1., "Forget bias for layer %s must be between 0-1" % layer_name
    assert layer_parameters["dropout_keep_prob"] >= 0. and layer_parameters["dropout_keep_prob"] <= 1., "dropout_keep_prob for layer %s must be between 0-1" % layer_name

    #store the output size for that layer. if bidirectional, store list with two copies of size so we can verify them later
    output_sizes[layer_name] = [ layer_parameters['hidden_size'], layer_parameters['hidden_size'] ] if layer_parameters['bidirectional'] else [ layer_parameters['hidden_size'] ] 

    #verify the dimensionality of the expected inputs at layer k equal the output dimensionalities from layer k-1 that connect via layer k's merge mode
    if cur_layer == 0:
      assert layer_parameters['expected_input_size'] == -1, "The expected_input_size of the first layer in the encoder must be -1. Instead it is %d" % layer_parameters['expected_input_size']
      assert len(layer_parameters['input_layers']) == 0, "Input layers for first layer in the encoder must be an empty list"
    else:
      assert "TODO" == "TODO", "write out little lambda function to add up layer input sizes based on merge mode"
      assert len(layer_parameters['input_layers']) > 0, "Input layers for all layers other than the first in the encoder must be a list with >1 elements"
      #assert they exist as keys

    cur_layer += 1

  assert encoder_json["num_layers"] == cur_layer, "Num layer property does not equal the length of the encoder layers"

  #TODO
  return True or False
  

def build_encoder_architecture_from_json(encoder_json_file_path):
  with open(encoder_json_file_path, 'rb') as model_data:
    try:
      encoder_model = json.load(model_data, object_pairs_hook=OrderedDict)
      print("Loaded JSON encoder model architecture from %s" % encoder_json_file_path)
      print("This architecture will now be verified...")

      if verify_encoder_architecture(encoder_model):
        print("Valid encoder architecture")
        return encoder_model #this is the JSON object parsed as a python dict
      else:
        print("Invalid encoder architecture")
        return False

    except ValueError, e:
      print("Invalid json in %s" % encoder_json_file_path)
      print(e)
      raise

  return False

def run_encoder_architecture(encoder_json):
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