from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import shape 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops #TODO - make this not so fucking old
from tensorflow.python.ops import nn_ops #TODO - make this not so fucking old

from collections import OrderedDict

#from tensorflow.contrib.rnn.python.ops import core_rnn
import custom_core_rnn as core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

#This is very similar to a dot product in this implementation between inputs and weights
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access
FLAGS = tf.app.flags.FLAGS #TODO - remove this dependency


def _create_decoder_cell(json_layer_parameters):
  if json_layer_parameters['lstm']:
    return _create_decoder_lstm(json_layer_parameters['hidden_size'],
                                json_layer_parameters['peepholes'],
                                json_layer_parameters['init_forget_bias'],
                                json_layer_parameters['dropout_keep_prob'])
  else:
    return _create_decoder_gru(json_layer_parameters['hidden_size'],
                                json_layer_parameters['peepholes'],
                                json_layer_parameters['init_forget_bias'],
                                json_layer_parameters['dropout_keep_prob'])

def _create_decoder_lstm(hidden_size, use_peepholes, init_forget_bias, dropout_keep_prob):
  c = core_rnn_cell_impl.LSTMCell(hidden_size, #number of units in the LSTM
                use_peepholes=use_peepholes,
                initializer=tf.contrib.layers.xavier_initializer(),
                forget_bias=init_forget_bias)
  if dropout_keep_prob < 1.0:
    c = core_rnn_cell_impl.DropoutWrapper(c, output_keep_prob=dropout_keep_prob)
  return c

def _create_decoder_gru(hidden_size, init_forget_bias, dropout_keep_prob):
  raise NotImplementedError

#This needs a lot more arguments...
#final_encoder_states - a list of lists containing 1 or 2 LSTM State Tuple/GRU States, depending on if that layer was bidirectional
#num_decoder_layers, an integer representing the depth of the stack of LSTMS/GRU's, but just for sanity until likely removed.
#returns the state for the decoder to start as a list itself, indexed by stack depth of the lstm or gru.
def initialize_decoder_states_from_final_encoder_states(final_encoder_states, decoder_architecture, decoder_state_initializer):

  #print("in initialize decoder states function. final encoder states is of type %s" % str(type(final_encoder_states)))
  #print("in initialize decoder states function. final encoder states -1 is of type %s" % str(type(final_encoder_states[-1])))
  #print("in initialize decoder states function. final encoder states -1 has length %d" % len(final_encoder_states[-1]))
  #print("in initialize decoder states function. final encoder states -1 has 0 element type %s" % str(type(final_encoder_states[-1][0])))
  
  if decoder_state_initializer == "top_layer_mirror":

    #will this won't necessarily work for all arbitrary geometries? i think so, UNLESS we mix GRU's with LSTM's
    #it also won't work if the top layer was bidirectional because we concatenated the states!

    #TODO
    #THIS WILL BREAK WITH BIDIRECTIONALITY IN THE TOP LAYER OF THE ENCODER OR ANYWHERE IN THE DECODER
    return [final_encoder_states[-1] for _ in xrange(decoder_architecture['num_layers'])] #a list of lists of LSTM State Tuples
  else:
    print("Haven't implemented the other cases yet for initializing decoder states.")
    raise NotImplementedError, "No other decoder state initialization has been written"
  
  #TODO - DO BETTER HERE and actually implement this

  #HERE IS WHAT WILL EVENTUALLY GO HERE

  #option to mimic encoder-decoder states if architectures are perfect mirror
  #option to just do what i did above, and use top layer status for all of them. still need parameterization
  #use a mean of the source annotation as per https://arxiv.org/pdf/1703.04357.pdf
  #


def _prepare_top_encoder_state(top_states):
#top states is a list of encoder states at the top layer of the encoder stack.
# this list might be a single tensor representing the state of the cell, or an lstm state tuple

  assert isinstance(top_states, (list)), "top states must be a list"
  is_lstm_tuple = top_states[0].__class__.__name__ == 'LSTMStateTuple' #otherwise it is a GRU

  #TODO - For now, this is more of a sanity check, but refactor this elsewhere
  if len(top_states) == 2:

    if is_lstm_tuple:

      assert top_states[0].__class__.__name__ == 'LSTMStateTuple'
      assert top_states[1].__class__.__name__ == 'LSTMStateTuple'
      #print("WARNING - top encoder layer is bidirectional, so two final states will be concatenated.")
      new_c = tf.concat([top_states[0][0], top_states[1][0]], axis=1 ) #combine c from both LSTMS
      #print("new C has shape %s" % str(new_c.get_shape()))
      new_h = tf.concat([top_states[0][1], top_states[1][1]], axis=1 ) #combine h from both LSTMS
      #print("new H has shape %s" % str(new_h.get_shape()))
      top_encoder_state = core_rnn_cell_impl.LSTMStateTuple(new_c, new_h)

    else:
      raise NotImplementedError, "Expected lstm state tuple. it isn't. gru not yet supported"
      top_encoder_state = tf.concat([top_states[0], top_states[1]], axis=1) #concat the bidirectional GRU hidden states
  
  elif len(top_states) == 1:
    top_encoder_state = top_states[0] #just a single lstm state tuple, so we're good
  else:
    raise ValueError("Too many states at top layer. Expected one or two. This isn't necessarily an error, but it is the case that there is no support yet written for an architecture that sees more than just a forward and backward state")

  return top_encoder_state




def _convolve_attention_states(reshaped_attention_states, num_attention_heads, attention_size):
  assert reshaped_attention_states.get_shape().with_rank(4)[3] == attention_size, "Reshaped attention states must have 4 dimensions. The last dimension of the reshaped attention states must be the same as the attention size"

  #these hidden features will be the reshaped attention states after being multiplied by a weight parameter, w1
  #in the attention model. there will be three sets of weights
  hidden_attention_states = []

  for attention_head_idx in xrange(num_attention_heads):

    #Here, a trick is used to use a 2d convolution after a reshape. we use the first set of weights in a convolutional network
    #This is W_1 in the model above. Notice there is a separate set of parameters for each of the possible attention heads
    attn_weights_1 = variable_scope.get_variable("attention_w1_head_%d" % attention_head_idx,
                                    [1, 1, attention_size, attention_size])

    #we build half of the term that is used for the attention mechanism calculation within the hyperbolic tangent 
    #(W_1 * attention_states)
    #we can do this outside of the attention mechanism call to save on some computation time
    hidden_attention_states.append(
      nn_ops.conv2d(reshaped_attention_states, attn_weights_1, [1, 1, 1, 1], "SAME")
    )

  return hidden_attention_states


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
    if output_projection is not None:
      prev = tf.add(tf.matmul(prev, output_projection[0]), output_projection[1])

    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


#Inputs:
# Decoder inputs: A list of 2d tensors with shape (batch_size, input_size). the length of the list is the bucket size.
# num_heads: an int, the number of attention heads
# attention_states: a 3d tensor with shape (batch_size, bucket_size, )
def validate_attention_decoder_inputs(decoder_inputs, num_heads, attention_states):

  print("Validate attention decoder inputs has been called. the shape of this muthafucka is %s" % attention_states.get_shape())

  if not decoder_inputs:
    raise ValueError("Your attention decoder has no decoder inputs")
  if num_heads < 1:
    raise ValueError("The number of heads to the attention decoder must be a positive integer. It is %d" % num_heads)
  if attention_states.get_shape().with_rank(3)[2].value is None:
    raise ValueError("The attention_size of the attention states must be known. Attention states come in the order (batch_size, attention_length, attention_size). This shape of this input now is %s" %
                     attention_states.get_shape())
  
  attn_length = attention_states.get_shape().with_rank(3)[1].value
  if attn_length is None:
    attn_length = array_ops.shape(attention_states)[1]

  attn_size = attention_states.get_shape().with_rank(3)[2].value

  return attn_length, attn_size


# this runs a 4-layer LSTM with residual connections from layers 0 -> 2, 0 -> 3, and 1 -> 3, and with no bidirectionality.
#
#parameters - 
#attn_input, Tensor, will be the attentive input to the decoder in all cases
#hidden states are the hidden states of the LSTM's at each layer in the stack. must be equal to the number of layers.
#scope is the scope with which to run this function, and for now is a keyword argument that I should probably clean up.
#
# returns - 
#
#TODO - modularize this to read from JSON like I did with image project
def decoder_rnn(attn_input, hidden_states_stack, num_layers=None):

  #print("in decoder rnn")
  ##print(type(hidden_states_stack))
  #print(type(hidden_states_stack[0]))
  #print(hidden_states_stack[0].get_shape())


  if num_layers != len(hidden_states_stack):
    raise ValueError("expected %d hidden states. instead only see %d hidden states" % (num_layers, len(hidden_states_stack)))

  with variable_scope.variable_scope('decoder_layer_0') as scope:
    fw0 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)
    outputs0, hidden_states_stack[0] = fw0(attn_input, hidden_states_stack[0], scope=scope)

    #print("at the end of the decoder_rnn layer 1, the scope name is %s" % scope.name)
  with variable_scope.variable_scope('decoder_layer_1') as scope:
    fw1 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)
    outputs1, hidden_states_stack[1] = fw1(outputs0, hidden_states_stack[1], scope=scope)
    #print("at the end of the decoder_rnn layer 2, the scope name is %s" % scope.name)  
  with variable_scope.variable_scope('decoder_layer_2') as scope:
    fw2 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)

    #add a residual connection to the outputs from the first layer
    #we don't need to call unstack here because the inputs into the __call__ function of the LSTMCell
    #are a single tensor, not a list of tensors.
    inputs2 = tf.add_n([outputs0,outputs1], name="residual_decoder_layer2_input")
    outputs2, hidden_states_stack[2] = fw2(inputs2, hidden_states_stack[2], scope=scope)
    #print("at the end of the decoder_rnn layer 3, the scope name is %s" % scope.name)
  with variable_scope.variable_scope('decoder_layer_3') as scope:
    fw3 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)

    #add a reisdual connection to the outputs from the second layer
    inputs3 = tf.add_n([outputs1, outputs2], name="residual_decoder_layer4_input")
    outputs3, hidden_states_stack[3] = fw3(inputs3, hidden_states_stack[3], scope=scope)
    #print("at the end of the decoder_rnn layer 4, the scope name is %s" % scope.name)
  output_size = fw3.output_size

  #we only care about the final output, but we need all the hidden cell states to pass back to this function later
  return outputs3, hidden_states_stack, output_size


def decoder_rnn_NEW(decoder_json, attn_input, hidden_states_stack, dtype=None):

#Runs a decoder RNN for use with an attention mechanism. Note this is different than the encoder RNN
# precisely because of this attention mechanism, and this is why we have the hidden states stack.
# this stack will be used so that we can pass in an initial state for the different layers in the 
# decoder stack when we are outputting.

  #print("\t\tDUDE, we're in the decoder now!")
  #print("\t\ttype of hidden states stack is %s" % str(type(hidden_states_stack)))
  #print("\t\ttype of first element in this stack is %s" % str(type(hidden_states_stack[0])))
  #print("\t\ttype of first element in the first list of this stack is %s" % str(type(hidden_states_stack[0][0])))
  #print("\t\tshape of this first LSTM state tuple element is %s" % str(hidden_states_stack[0][0][0].get_shape()))
  #print("\t\tshape of this second LSTM state tuple element is %s" % str(hidden_states_stack[0][0][0].get_shape()))
  with variable_scope.variable_scope("decoder", dtype=dtype) as scope:

    current_layer = 0 #not needed really
    cell_outputs = OrderedDict()
    output_size = None
    #new_cell_states = OrderedDict() #probably dont need this, write over hidden_states_stack

    for layer_name, layer_parameters in decoder_json["layers"].iteritems(): #this is an ordered dict, so this is okay
      #print("\t\tdecoder is analyzing layer %d" % current_layer)

      with variable_scope.variable_scope(layer_name) as scope:
        #print("\t\tdecoder is currently at scope %s" % scope.name)

        input_list = []

        #add to inputs the residual connections that connect to this layer
        for candidate_layer_name, candidate_layer_output in cell_outputs.iteritems():
          #print("layer %s is checking layer %s for inputs" %  (layer_name, candidate_layer_name))
          if candidate_layer_name == layer_name: #layers don't have residual connections to themselves...
            #print("skipping")
            continue
          elif candidate_layer_name in layer_parameters['input_layers']: #we will gather the layers that are marked in the architecture
            #print("layer %s will use input from layer %s" % (layer_name, candidate_layer_name))
            for layer_output in candidate_layer_output: #candidate_layer_output is always a list, sometimes with 1 element, sometimes with 2
              input_list.append(layer_output)

        #print("\t\tlayer %s has %d total inputs in the input list." % (layer_name, len(input_list)))

        #combine these residual inputs following this layer's merge mode
        if layer_parameters['input_merge_mode'] == 'concat':
          #inputs = tf.cond(len(input_list) > 1,
          #  lambda:tf.concat(input_list, axis=1),
          #  lambda:input_list[0])
          #print("\t\tthe shape of the input list BEFORE concatenating is shape %s and there are %d of them" % (str(inputs[0].get_shape()), len(input_list)))
          inputs = tf.unstack(tf.concat(input_list, axis=2)) #we unstack to get a list, but needed for ONE time step?
          #print("\t\tthe shape of the inputs AFTER concatenating is %s and there are %d of them" % (str(inputs[0].get_shape()), len(inputs)))
        elif layer_parameters['input_merge_mode'] == 'sum':
          #inputs = tf.unstack(tf.add_n([i for i in input_list])) #TODO - correct here? unstack check
          #inputs = tf.cond(len(input_list) > 1,
          #  lambda:tf.unstack(tf.add_n(input_list)),
          #  lambda:input_list[0])
          inputs = tf.unstack(tf.add_n(input_list)) #NEED TO UNSTACK FOR ONE TIME STEP?
          #print("\t\tgoing to sum inputs into layer %s\nThe dimensionality of the first tensor in the input list is %s" % (layer_name, str(inputs[0].get_shape())))
          #print("\t\tthe shape of the inputs after summing is %s and there are %d of them" % (str(inputs[0].get_shape()), len(inputs)))
        else:
          assert current_layer==0, "The input merge mode must be concat or sum for all other layers but 0" #remove this text
          assert len(input_list)==0, "The current layer should be layer 0 if there are no inputs from other layers" #remove this test
          #print("\t\tthe attentive input has shape %s" % (str(attn_input.get_shape())))
          inputs = [attn_input]

        #print("\t\tAbout to run the network layer. Inputs have shape %s and length %d" % (str(inputs[0].get_shape()), len(inputs)))

                #create/get the cells, and run them
        #this will give us the outputs, and we can combine them as necessary
        #c stands for cell, out for outputs, f for forward, b for backward
        if layer_parameters['bidirectional']:
          assert isinstance(hidden_states_stack[current_layer], (list)), "Current layers hidden states stack must be a list because it is bidirectional"
          assert len(hidden_states_stack[current_layer]) == 2, "Expected current layer to have two initial hidden states, but instead have %d" % len(hidden_states_stack)
          cf = _create_decoder_cell(layer_parameters)
          cb = _create_decoder_cell(layer_parameters)
          out_fb, out_f, out_b, state_f, state_b = core_rnn.static_bidirectional_rnn(cf,
                                                                                    cb,
                                                                                    inputs, #ONE TIME STEP
                                                                                    initial_state_forward=hidden_states_stack[current_layer][0],
                                                                                    initial_state_backward=hidden_states_stack[current_layer][1],
                                                                                    dtype=dtype)
          #print("\t\tbidirectional rnn ran.\n\ttype of out_fb is %s\n\ttype of out_f is %s\n\ttype of out_b is %s\n\ttype of state_f is %s\n\ttype of state_b is %s" % (str(type(out_fb)), str(type(out_f)), str(type(out_b)), str(type(state_f)), str(type(state_b))))
          #print("shape of out_forward for layer %s is %s " % (layer_name, str(out_f.get_shape())))

          #store the outputs according to how they have to be merged.
          #they will be a list with 2 elements, the forward and backward outputs. or, a list with one element, the concatenation or sum of the 2 elements.
          
          #TODO - remove these checks form every encoder pass
          if layer_parameters['output_merge_mode'] == 'concat':
            #assert out_f[0].get_shape().ndims == 2
            cell_outputs[layer_name] = [out_fb]
          elif layer_parameters['output_merge_mode'] == 'sum':
            #assert out_f[0].get_shape().ndims == 2
            cell_outputs[layer_name] = [tf.unstack(tf.add_n([out_f, out_b]))] #unstack recreates a list of length bucket_length of tensors with shape (batch_size, hidden_size)
          else:
            cell_outputs[layer_name] = [out_f, out_b]

          #out_f is a list, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          hidden_states_stack[current_layer] = [state_f,state_b]

        else:
          cf = _create_decoder_cell(layer_parameters)

          #out_f is a list of tensor outputs, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          out_f, state_f = core_rnn.static_rnn(cf, inputs, initial_state=hidden_states_stack[current_layer][0], dtype=dtype) #ONE TIME STEP
          #print("\t\tunidirectional rnn ran.\n\ttype of out_f is %s\n\ttype of state_f is %s" % (str(type(out_f)),str(type(state_f))))
          cell_outputs[layer_name] = [out_f]
          hidden_states_stack[current_layer] = [state_f]

      top_layer_name = layer_name # just for readability
      current_layer += 1

    #We do this anyway outside of the if statement, because the concatenation won't do anything otherwise.
    #if len(cell_outputs[top_layer_name]) > 1:
    #  print("\t\ERROR - your top layer cell outputs more than a single tensor at each time step. Perhaps it is bidirectional with no output merge mode specified. These tensors will be concatenated along axis 1. You should change this in the JSON to be 'concat' for readability, or 'sum' if you want the tensors element-wise added.")
    stack_output = tf.unstack(tf.concat(cell_outputs[top_layer_name], axis=1))

    #we only care about the final output, but we need all the hidden cell states to pass back to this function later
    return stack_output, hidden_states_stack

#Inputs:
#
#batch_size - int, the batch size dimension value from the decoder inputs
#attention_size - int, chosen attention mechanism hidden size
#num_attn_heads - int, the value of the flag for the number of attention heads to use
#
#returns - a list of tensors, shaped batch_size, attn_size, filled with 0 values. list has length num_attn_heads
def initialize_attention(batch_size,
                        attn_size,
                        num_attn_heads,
                        dtype=None):

  #the size of the batch attention will be batch_size x attention size, and we stack these values along the 0 axis
  batch_attn_size = tf.stack([batch_size, attn_size])
  
  attns = [
      array_ops.zeros(
          batch_attn_size, dtype=dtype) for _ in xrange(num_attn_heads)
  ]

  # Make sure the attention size is properly set.
  for attn in attns:
    attn.set_shape([None, attn_size])

  return attns

  

def run_manning_attention_mechanism():
  raise NotImplementedError()
  #TODO - write this. simple scoring function of h_t * (W_s * h_s). 1 set of weights vs. 3 in bahdanu's


def run_bahdanu_attention_mechanism(query_state,
  reshaped_attention_states,
  hidden_attention_states,
  weights_v,
  num_attention_heads,
  attention_size,
  attention_length):

  attention_reads = [] #This will be built dynamically as we read through the attention head

  #print(query_state.__class__.__name__)
  #print(type(query_state))
  #print(query_state[0].get_shape())
  #print(type(query_state[0]))
  #print(query_state[0].name)
  #print(query_state[1].get_shape())
  #print(type(query_state[1]))
  #print(query_state[1].name)

  assert query_state.__class__.__name__ == 'LSTMStateTuple', "The decoder state passed to the attention model must be an LSTMStateTuple (c,h)"

  #print("size of first element of lstm state tuple in attention is " + str(query_state[0].get_shape()))
  #print("size of second element of lstm state tuple in attention is " + str(query_state[1].get_shape()))

  #let's verify that the cell state and the hidden state
  assert len(hidden_attention_states) == num_attention_heads, "There must be the same number of calculated hidden attention states from the 1x1 convolution as there are number of attention heads."

  #now we concatenate the two states of the lstm across the second dimension (the one that is not the batch size) 
  #so that we may run the attention model on it
  #notice that this means we are concatenating the LSTM cell state and the lstm hidden state across one dimension
  query_state = array_ops.concat(nest.flatten(query_state), 1)

  #print(query_state.get_shape())
  #print("The attention mechanism has flattened the query shape to " + str(query_state.get_shape()))

  for head_idx in xrange(num_attention_heads):
    with variable_scope.variable_scope("Attention_%d" % head_idx):

      #print("applying linear transformation with attention vector size " + str(attention_vec_size))

      #we apply a linear transformation to the (batch_size, decoder_hidden_size*2) query state to 
      #transform it into the size of our attention vector)
      #this gives us another set of weights, namely weights_2, or U in the above definition
      weighted_new_state = linear(query_state, attention_size, True)

      #print("after transformation, new shape is " + str(weighted_new_state.get_shape()))

      #this vector needs to be reshaped into the same shape as the hidden features.
      weighted_new_state = array_ops.reshape(weighted_new_state, [-1, 1, 1, attention_size])

      #print("after reshape squeeze, new shape is " + str(weighted_new_state.get_shape()))
      
      # Attention mask is a softmax of V^T * tanh(W_1 * attention_states + W_2 * new_state)
      # The first term in the tangent function is W_1*attention_states, and the second term is U*new_state
      # We reduce the sum over multiple heads of attention because we may use more than 1. The indexes 2 and 3
      # are responsible for getting rid of the convolutional dimensional expansion we did earlier.
      s = math_ops.reduce_sum(weights_v[head_idx] * math_ops.tanh(hidden_attention_states[head_idx] + weighted_new_state),
                              [2, 3])

      #print("after attention hyperbolic tangent function, new shape is " + str(s.get_shape()))
      softmax_attention_weights = nn_ops.softmax(s)
      #print("after softmax function, new shape is " + str(softmax_attention_weights.get_shape()))

      # Now calculate the attention-weighted vector d.
      weighted_attention = math_ops.reduce_sum(
          array_ops.reshape(softmax_attention_weights, [-1, attention_length, 1, 1]) * reshaped_attention_states, [1, 2])

      #print("after attention weighted representation of vector, new shape is " + str(weighted_attention.get_shape()))
      attention_reads.append(array_ops.reshape(weighted_attention, [-1, attention_size]))
      #print("this representation has been reshaped to the final shape of " + str(attention_reads[-1].get_shape()))
  return attention_reads


def attention_decoder(decoder_architecture,
                      decoder_state_initializer,
                      decoder_inputs,
                      final_encoder_states, #This is a LIST
                      attention_states,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  Implementation based on http://arxiv.org/abs/1412.7449 

  Rob Documentation:
    Arguments:

    decoder-inputs: This is a list of 2D Tensors of shape [batch_size, input_size].
      Depending on the bucket size, input_size will be a padded sequence of whatever
      length the target language bucket size is. 

    final_encoder_states: The initial decoder state will be calculated from this.
      This will be a list of LSTMStateTuples or GRU States, and this list might well
      only have one element if the network is one layer deep. The indexes refer
      to the cell depths, ie, final_encoder_state[1] would be an LSTMStateTuple for layer 2's
      lstm cell state in the final layer of the encoder. If using GRU's, then final_encoder_states[4]
      would be the final hidden state for the 5th layer GRU in the encoder.

  
  Argumentss:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """

  assert isinstance(final_encoder_states, (list)), "Final encoder states must be a list"

  #TODO - introduce manning's attention mechanism that scores h_t * (W*h_s) instead of this version
  #THIS REQUIRES A REFACTOR
  #Bahdanu attention has more trainable weights (3 sets) compared to Manning's (1 set of weights) - see cs224n lec 16
  with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:

    #TODO - some assertion about the lstm state tuple only being there if we are indeed dealing with an lstm, not gru
    #print("Reached the attention decoder. The type of the initial state is %s" % str(type(final_encoder_states)))

    #verify we have known shapes and nonzero inputs
    attn_length, attn_size = validate_attention_decoder_inputs(decoder_inputs, num_heads, attention_states)

    #print("called attention decoder. current scope is %s" % scope.name)

    #store a scope reference as well as a datatype reference for book-keeping and debugging
    restore_scope = scope
    dtype = scope.dtype

    #print("the attention decoder has been called with attention states sized " + str(attention_states.get_shape()))

    #the attention states come in and they are shaped (probably) as such:
    #   (batch size, attn_length (bucket length, or 1 if decoding), attention_size)
    # these need to be reshaped for a convolutional operation that represents the 
    reshaped_attention_states = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])

    #print("reshaped attention states have shape %s" % str(reshaped_attention_states.get_shape()))=
    #print("the attention decoder has reshaped the attention state to size " + str(reshaped_attention_states.get_shape()))

    #we need to construct the following equation:
    # attention = softmax(V^T * tanh(W_1 * attention_states + W_2 * new_state)), where new_state is produced on each cell output
    # there are three parameters here, namely, V^t, W_1 and W_2. 
    
    #we will use bahdanu attention.
    #store the attention parameter V
    weights_v = []
    for head_idx in xrange(num_heads):
      weights_v.append(
          variable_scope.get_variable("attention_v_head_%d" % head_idx, [attn_size])
      )

    #print("the attention decoder has reshaped the attention state to a new size using 1x1 convolution.
    #now size is " + str(hidden_attention_states[head_idx].get_shape()))
    
    #we need to store the batch size of the decoder inputs to use later for reshaping.
    # because these come in as a list of tensors, just take the first one.
    # this will store a TENSOR of one-dimension with the batch_size shape.

    #the batch size is inferred from the list 2d-shaped (batch_size, input_size) decoder_inputs
    attentions = initialize_attention( tf.shape(decoder_inputs[0])[0], attn_size, num_heads, dtype=dtype)
  
    #store the attention parameter W_1 * attention_state
    hidden_attention_states = _convolve_attention_states(reshaped_attention_states, num_heads, attn_size)

    if initial_state_attention:

      #This is a lot of work for initial state attention, maybe we should just tell this feature to fuck off
      #TODO = eventually this should get abstracted away probably
      top_encoder_state_list = final_encoder_states[-1]
      reshaped_top_encoder_state = _prepare_top_encoder_state(top_encoder_state_list)

      attentions = run_bahdanu_attention_mechanism(reshaped_top_encoder_state,
                                      reshaped_attention_states,
                                      hidden_attention_states,
                                      weights_v,
                                      num_heads,
                                      attn_size,
                                      attn_length)

    #We need to initialize the hidden states of the decoder before we start putting inputs into the network
    #This can be done in many ways, so we call this off to another function.
    # however, in all circumstances, we are just trying to steal some domain knowledge from the final encoder
    # states, but how to choose other parameters, hoofta.
    decoder_hidden_states = initialize_decoder_states_from_final_encoder_states(final_encoder_states,
                                                                        decoder_architecture,
                                                                        decoder_state_initializer)

    #this is the initial value for the loop counters that track the outputs at each time step, 
    # as well as a second reference to the previous output, for readability
    previous_decoder_output = None
    outputs = []

    #=============================Main Decoder Loop================================
    for decoder_time_step, decoder_input in enumerate(decoder_inputs):

      #we only set reuse to true after the first run in the loop. they'll only get changed one time, on the first run
      # in tensorflow, on the iteration after they're initially created in memory. and that's good.
      if decoder_time_step > 0:
        variable_scope.get_variable_scope().reuse_variables()
      

      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and previous_decoder_output is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          print("\t\t\tthe loop function is not none")
          decoder_input = loop_function(previous_decoder_output, decoder_time_step)

      #TODO - refactor this
      # Merge input and previous attentions into one vector of the right size.
      #TODO - with rank should probably be used throughout this file a lot more
      input_size = decoder_input.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % decoder_input.name)

      #attentions is a list, probably of length one because num_heads is probably one, but it is still a list.
      #decoder_input is not, it's a tensor, so we put it in a list.
      attentive_input = linear([decoder_input] + attentions, input_size, True)


      #TODO - output_size is only written for the unidirectional case under the presumption the top decoder layer won't be bidirectional
      #decoder_output, hidden_states, output_size = decoder_rnn(attentive_input,
      #                                                        hidden_states,
      #                                                        num_layers=num_decoder_layers)

      #This is similar to the encoder mechanism we wrote earlier but the main difference
      # (other than the whole-attention-mechanism thing) is that this decoder call runs for ONE time step
      #decoder output is a list with a single tensor. eventually this list could be removed
      #decoder hidden states is a list of lists of tensors
      decoder_output, decoder_hidden_states = decoder_rnn_NEW(decoder_architecture,
                                                              attentive_input,
                                                              decoder_hidden_states,
                                                              dtype=dtype)

      #this can probably go somewhere else. added with_rank(2)
      output_size = decoder_output[0].get_shape().with_rank(2)[1]

      #hidden states is a list of LSTMStateTuples which each store the states for each layer of the decoder stack
      #hidden_states[0] - lstmstatetuple storing (cell_state, hidden_state) for lowest layer of lstm stack
      #hidden_states[1] = lstmstatetuple storing .. for layer 1 of lstm stack
      #hidden_states[-1] = lstmstatetuple storing top layer of lstm stack

      #TODO - this probably needs to become [-1][0] because this is a list of lists, so this assertion should catch it
      #print("I am looking for a list. List element zero is type %s" % str(hidden_states[-1][0].__class__.__name__))
      assert decoder_hidden_states[-1].__class__.__name__ == 'list', "Decoder hidden state's elements are lists."
      assert len(decoder_hidden_states[-1]) == 1, "decoder hidden state at top layer needs to only have one lstmstatetuple in the list. this is because we pass it directly as the query to the bahdanu attention mechanism, which expects an lstm tuple"
      assert decoder_hidden_states[-1][0].__class__.__name__ == 'LSTMStateTuple', "Expected hiddenstates -1 to have an lstm tuple. Instead it is %s" % decoder_hidden_states[-1].__class__.__name__
      
      #still decoder state will be an LSTMStateTuple
      top_decoder_state = decoder_hidden_states[-1][0]

      # Run the attention mechanism.
      # Notice how we still call hidden attention states. the reason for this is that we calcualted this beforehand and it
      # does not change. what changes is the decoder state that is half of the built in term within the hyperbolic tangent activation function.
      if decoder_time_step == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attentions = run_bahdanu_attention_mechanism(top_decoder_state, reshaped_attention_states, hidden_attention_states, weights_v, num_heads, attn_size, attn_length)
      else:
        attentions = run_bahdanu_attention_mechanism(top_decoder_state, reshaped_attention_states, hidden_attention_states, weights_v, num_heads, attn_size, attn_length)

      #print("after calling attention mechanism, the output shape of the first item in the attns list is " + str(attns[0].get_shape()))

      #Now that we have all the pieces for the output from the decoder, we have one final parameter to train,
      # namely, the weight and bias vector that will be used to transform the dimensionality of our output
      # from the decoder into the dimensionality specified by the output projection
      with variable_scope.variable_scope("output_projection"):

        #the attentions are a list, and the decoder output is a list as well, so we can pass these directly to linear
        decoder_output = linear(decoder_output + attentions, output_size, True)

      #Append out final outputs to the list, adjust the loop iterator, and continue decoding...  
      if loop_function is not None:
        previous_decoder_output = decoder_output
      outputs.append(decoder_output)
    #================================End Main Decoder Loop===================================

  return outputs, top_decoder_state



def embedding_attention_decoder(decoder_architecture,
                                decoder_state_initializer,
                                decoder_inputs,
                                final_encoder_states,
                                attention_states,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:

    decoder_architecture: Python-parsed JSON object of the decoder read from encoder_decoder_architecture.json

    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).

    final_encoder_states: A list of encoder states. Index i corresponds to the encoder state at layer i in the
     encoder. Index -1 is therefore the top layer state index at the final time step. At index i, the list will
     contain another list. The elements in this list depend on the architecture. They will be either:
      1. A single 2d tensor of size [batch_size, cell_state_size]
      2. A single LSTM tuple, with each tuple being a single 2d tensor of size [batch_size, cell_state_size]
      3. Two LSTM tuples, in case the top layer is bidirectional, with each tuple being a single 2d tensor of shape [batch_size, cell_state_size]

    attention_states: 3D Tensor [batch_size x attn_length x attn_size]
    The attention length is most likely going to be one. The attention size will be whatever the output size of
    the LSTM was at the top layer of the encoder. If this layer is bidirectional, then it will be a concatenation
    of these two outputs.

    num_symbols: The number of possible words in the embedding.

    embedding_size: Integer, the length of the embedding vector for each word.

    num_heads: Number of attention heads that read from attention_states.

    output_size: Size of the output vectors; if None, use output_size.

    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.

    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).

    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.

    dtype: The dtype to use for the RNN initial states (default: tf.float32).

    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".

    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  #if output_size is None:
  #  output_size = cell.output_size

  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])


    #If we are actively decoding, then we don't care about the decoder inputs because we generate them
    # via a feed_previous option
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None

    #The embedding inputs represent the lookups in the embedding table for each symbol in the decoder, which we 
    # may or may not use, depending on if we are training or actively decoding. 
    embedding_inputs = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
    ]

    return attention_decoder(
        decoder_architecture,
        decoder_state_initializer,
        embedding_inputs,
        final_encoder_states, #TODO this is a LIST, make changes accordingly
        attention_states,
        output_size=None,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)