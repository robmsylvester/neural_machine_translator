from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import model_utils
import embeddings

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import shape 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from collections import OrderedDict
import custom_core_rnn as core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

#This is very similar to a dot product in this implementation between inputs and weights

#TODO - replace linear program wide with with a fully connected layer call at 
from tensorflow.contrib.layers import fully_connected
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access


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


#NOTE - Bahdanu attention for deep models adds a lot of parameters, so do this only if you have a lot of time
# and a good gpu or two or ten
def _initialize_decoder_states_bahdanu(final_encoder_state_tensor, decoder_architecture, is_lstm):
  #Args:
  # 1. final_encoder_state_tensor - the final encoder state tensor is a single tensor representing the values of the states at the final time step
  #  for the top layer in the encoder. it has dimension (batch_size, hidden_size). in the event that the final layer
  #  is bidirectional, this will only be the BACKWARD cell states, so still a tensor of size (batch_size, hidden_size)
  #  if is_lstm is true, then this is an lstmstatetuple, which stores two tensors of the dimensionality (batch_size, hidden_size.
  # 2. decoder architecture - json object
  # 3. is_lstm - whether to break apart the tuple and use two weights or just one in the event that you are using gru's
  #Returns:
  # A python list of lists, of length decoder_architecture[num_layers], where each list has length 1 or 2, depending on if that layer is bidirectional
  #  these lists contain tensors to use as the initial values for the lstms in the decoder
  #

  initial_decoder_state_tensors = [] #list of lists

  for layer_idx, layer_parameters in decoder_architecture['layers'].iteritems():

    if is_lstm:
      cell_state,hidden_state = final_encoder_state_tensor
      
      dec_cell_state = fully_connected(cell_state,
                                      layer_parameters["hidden_size"],
                                      activation_fn=None, #linear activation for states
                                      weights_initializer=initializers.xavier_initializer(), #or do i want random uniform?
                                      biases_initializer=init_ops.zeros_initializer(),
                                      reuse=True,
                                      scope="dec_%s_lstm_cell_state_init_linear"%layer_idx)

      dec_hidden_state = fully_connected(cell_state,
                                layer_parameters["hidden_size"],
                                activation_fn=None, #linear activation for states
                                weights_initializer=initializers.xavier_initializer(), #or do i want random uniform?
                                biases_initializer=init_ops.zeros_initializer(),
                                reuse=True,
                                scope="dec_%s_lstm_hidden_state_init_linear"%layer_idx)

      dec_state = core_rnn_cell_impl.LSTMStateTuple(dec_cell_state, dec_hidden_state)

    else: #is gru
      raise NotImplementedError, "GRU support has not been implemented"
      dec_state = fully_connected(final_encoder_state_tensor,
                            layer_parameters["hidden_size"],
                            activation_fn=None, #linear activation for states
                            weights_initializer=initializers.xavier_initializer(), #or do i want random uniform?
                            biases_initializer=init_ops.zeros_initializer(),
                            reuse=True,
                            scope="dec_%s_gru_state_init_linear"%layer_idx)

    #store the tensors in a list of lists
    initial_decoder_state_tensors.append( [dec_state, dec_state] if layer_parameters["bidirectional"] else [dec_state] )

  return initial_decoder_state_tensors


      

#final_encoder_states - a list of lists containing 1 or 2 LSTM State Tuple/GRU States, depending on if that layer was bidirectional
#num_decoder_layers, an integer representing the depth of the stack of LSTMS/GRU's, but just for sanity until likely removed.
#returns the state for the decoder to start as a list itself, indexed by stack depth of the lstm or gru.
def initialize_decoder_states_from_final_encoder_states(final_encoder_states, decoder_architecture, decoder_state_initializer):

  #option to mimic encoder-decoder states if architectures are perfect mirror
  #option to just do what i did above, and use top layer status for all of them. still need parameterization
  #use a mean of the source annotation as per https://arxiv.org/pdf/1703.04357.pdf
  
  if decoder_state_initializer == "mirror":
    return final_encoder_states #the encoder states will be used 1-for-1 in the decoder architecture because it's exactly the same

  elif decoder_state_initializer == "top_layer_mirror":
    
    #this won't necessarily work for all arbitrary geometries? i think its okay, UNLESS we mix GRU's with LSTM's
    #it also won't work if the top layer was bidirectional because we concatenated the states!
    if len(final_encoder_states[-1] == 2):
      combined_lstm_tuple = _concatenate_bidirectional_lstm_state_tuples(final_encoder_states[-1][0], final_encoder_states[-1][1])    
      return [ [combined_lstm_tuple] for _ in xrange(decoder_architecture['num_layers'])] #a list of lists of LSTM State Tuples
    else:
      return [final_encoder_states[-1] for _ in xrange(decoder_architecture['num_layers'])] #a list of lists of LSTM State Tuples
  
  elif decoder_state_initializer == "nematus":
    raise NotImplementedError, "Nematus intializer is not implemented. To do this. All encoder states need to be tracked or averaged in the encoder pass"
  
  elif decoder_state_initializer == "bahdanu": #only take the backward bidirectional states, if they're there.
    return _initialize_decoder_states_bahdanu(final_encoder_states[-1][-1], decoder_architecture, is_lstm=True) #TODO - remove this True statement when GRU implemented
  else:
    raise NotImplementedError, "No other decoder state initialization has been written yet."



#Simple function to combine the cell states and hidden states for multiple LSTMS into a single concatenated tensor
def _concatenate_bidirectional_lstm_state_tuples(lstm_fw_state_tuple, lstm_bw_state_tuple):
  assert lstm_state_tuple_1.__class__.__name__ == 'LSTMStateTuple'
  assert lstm_state_tuple_2.__class__.__name__ == 'LSTMStateTuple'

  new_c = tf.concat([lstm_fw_state_tuple[0], lstm_bw_state_tuple[0]], axis=1 ) #combine c from both LSTMS
  new_h = tf.concat([lstm_fw_state_tuple[1], lstm_bw_state_tuple[1]], axis=1 ) #combine h from both LSTMS
  return core_rnn_cell_impl.LSTMStateTuple(new_c, new_h)



#TODO - remove dec_state_initializer argument
def _prepare_top_lstm_encoder_state_for_attention(top_states):
  """
  Takes a look at the top states of the encoder stack, which should be LSTM State Tuples
  There is probably just one LSTM state tuple, but there will be two if they are bidirectional at the top layer of encoder.
  If they are not bidirectional, just return the LSTM
  If they are bidirectional, concatenate the cell states and hidden states of the two tuples together and return a single
   LSTMStateTuple of size (2x cell state, 2x hidden state)

  #Parameters:
  #top states is a list of encoder states at the top layer of the encoder stack.
  # this list might be a single lstm state tuple, or a list of lstm state tuples
  # decoder_state_initializer - depending on the strategy used in the literature, you do different things with these possible states

  #Returns:
   LSTM state tuple.
  """
  assert isinstance(top_states, (list)), "top states must be a list"
  assert top_states[0].__class__.__name__ == 'LSTMStateTuple', "Expected LSTM State Tuple. Got %s" % top_states[0].__class__.__name__#otherwise it is a GRU

  #TODO - For now, this is more of a sanity check, but refactor this elsewhere
  if len(top_states) == 2:
      top_encoder_state = _concatenate_bidirectional_lstm_state_tuples(top_states[0], top_states[1])
  
  elif len(top_states) == 1:
    top_encoder_state = top_states[0] #just a single lstm state tuple, so we're good
  else:
    raise ValueError("Too many states at top layer. Expected one or two. This isn't necessarily an error, but it is the case that there is no support yet written for an architecture that sees more than just a forward and backward state")

  return top_encoder_state



#One of the ways that attention state are reshaped is to create a 1x1 convolution between the reshaped attention states
# and the attention weights. This is opposed to a standard feedforward layer.
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
      tf.nn.conv2d(reshaped_attention_states, attn_weights_1, [1, 1, 1, 1], "SAME")
    )

  return hidden_attention_states


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Argumentss:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair of weights, biases). previous outputs are multiplied by weights and the biases are added
    update_embedding: Boolean; if False, the gradients will not propagate through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):

    #Reshape output projection
    if output_projection is not None:
      prev = tf.add(tf.matmul(prev, output_projection[0]), output_projection[1])

    #Get the most likely word
    prev_symbol = tf.argmax(prev, 1)

    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.

    #Look up the previous symbol in the embedding table
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)

    #Kill the partial derivative calculation if we don't need it (such as when we are decoding or doing something with validation sets)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev

  return loop_function



def validate_attention_decoder_inputs(decoder_inputs, num_heads, attention_states):
#Helper function that will make sure our dimensions are correct for the attention decoder inputs

#Args:
# Decoder inputs: A list of 2d tensors with shape (batch_size, input_size). the length of the list is the bucket size.
# num_heads: an int, the number of attention heads
# attention_states: a 3d tensor with shape (batch_size, bucket_size, input_size)

#Returns:
# Tuple - the attention length and the attention size. The attention length is probably one.

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


def one_step_decoder(decoder_json, attn_input, input_lengths, hidden_states_stack, dtype=None):

#Runs a decoder RNN for use with an attention mechanism. Note this is different than the encoder RNN
# precisely because of this attention mechanism, and this is why we have the hidden states stack.
# this stack will be used so that we can pass in an initial state for the different layers in the 
# decoder stack when we are outputting.

  #sanity checks if you want them
  #print("\t\ttype of hidden states stack is %s" % str(type(hidden_states_stack)))
  #print("\t\ttype of first element in this stack is %s" % str(type(hidden_states_stack[0])))
  #print("\t\ttype of first element in the first list of this stack is %s" % str(type(hidden_states_stack[0][0])))
  #print("\t\tshape of this first LSTM state tuple element is %s" % str(hidden_states_stack[0][0][0].get_shape()))
  #print("\t\tshape of this second LSTM state tuple element is %s" % str(hidden_states_stack[0][0][0].get_shape()))
  with variable_scope.variable_scope("decoder", dtype=dtype) as scope:

    current_layer = 0
    cell_outputs = OrderedDict()
    output_size = None
    #new_cell_states = OrderedDict() #probably dont need this, write over hidden_states_stack

    for layer_name, layer_parameters in decoder_json["layers"].iteritems(): #this is an ordered dict, so this is okay
      #print("\t\tdecoder is analyzing layer %d" % current_layer)

      with variable_scope.variable_scope(layer_name) as scope:
        #print("\t\tdecoder is currently at scope %s" % scope.name)


        #first loop will get embeddings, later loops will use previous iteration outputs, plus any residual connections
        input_list = model_utils._get_residual_layer_inputs_as_list(layer_name, layer_parameters['input_layers'], cell_outputs)

        #combine these inputs according to sum/concat, unless there are no residual inputs, in which case we are in layer 1 and use attentive input
        #we dont return a list because we pass this to a dynamic decoder, which is just one tensor and uses the time major axis instead of a list of length max time
        inputs = model_utils._combine_residual_inputs(input_list, layer_parameters['input_merge_mode'], return_list=True) if len(input_list) else [attn_input]

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
  
  #the attentions are a list, of length attention heads, that begin at zero
  attns = [ tf.zeros(batch_attn_size, dtype=dtype) for _ in xrange(num_attn_heads)]

  # Make sure the attention size is the second dimension, where batch size is the first)
  for attn in attns:
    attn.set_shape([None, attn_size])

  return attns

  

def run_manning_attention_mechanism():
  raise NotImplementedError()
  #TODO - write this. simple scoring function of h_t * (W_s * h_s). 1 set of weights vs. 3 in bahdanu's


#TODO - add is_lstm argument for gru support when added
def run_bahdanu_attention_mechanism(query_state,
                                    reshaped_attention_states,
                                    hidden_attention_states,
                                    weights_v,
                                    num_attention_heads,
                                    attention_size,
                                    attention_length):

  attention_reads = [] #This will be built dynamically as we read through the attention head

  #if is_lstm:
  assert query_state.__class__.__name__ == 'LSTMStateTuple', "The decoder state passed to the attention model must be an LSTMStateTuple (c,h)"
  #else assert gru

  #let's verify that the cell state and the hidden state
  assert len(hidden_attention_states) == num_attention_heads, "There must be the same number of calculated hidden attention states from the 1x1 convolution as there are number of attention heads."

  #now we concatenate the two states of the lstm across the second dimension (the one that is not the batch size) 
  #so that we may run the attention model on it
  #notice that this means we are concatenating the LSTM cell state and the lstm hidden state across one dimension
  query_state = tf.concat(nest.flatten(query_state), 1) #axis=1

  for head_idx in xrange(num_attention_heads):
    with variable_scope.variable_scope("Attention_%d" % head_idx):

      #we apply a linear transformation to the (batch_size, decoder_hidden_size*2) query state to 
      #transform it into the size of our attention vector)
      #this gives us another set of weights, namely weights_2, or U in the above definition
      weighted_new_state = fully_connected(query_state,
                                           attention_size,
                                           activation_fn=None,
                                           weights_initializer=initializers.xavier_initializer(),
                                           biases_initializer=init_ops.zeros_initializer(),
                                           reuse=True,
                                           scope="w_%d" % head_idx)

      #print("after transformation, new shape is " + str(weighted_new_state.get_shape()))

      #this vector needs to be reshaped into the same shape as the hidden features.
      weighted_new_state = array_ops.reshape(weighted_new_state, [-1, 1, 1, attention_size])

      #print("after reshape squeeze, new shape is " + str(weighted_new_state.get_shape()))
      
      # Attention mask is a softmax of V^T * tanh(W_1 * attention_states + W_2 * new_state)
      # The first term in the tangent function is W_1*attention_states, and the second term is U*new_state
      # We reduce the sum over multiple heads of attention because we may use more than 1. The indexes 2 and 3
      # are responsible for getting rid of the convolutional dimensional expansion we did earlier by specifying
      # these two axis that we are aiming to reduce.
      s = tf.reduce_sum(weights_v[head_idx] * tf.tanh(hidden_attention_states[head_idx] + weighted_new_state),
                              [2, 3])

      softmax_attention_weights = tf.nn.softmax(s)

      # Now calculate the attention-weighted vector d.
      weighted_attention = tf.reduce_sum(
          array_ops.reshape(softmax_attention_weights, [-1, attention_length, 1, 1]) * reshaped_attention_states, [1, 2])

      attention_reads.append(array_ops.reshape(weighted_attention, [-1, attention_size]))
      #print("this representation has been reshaped to the list with elements that have shape of " + str(attention_reads[-1].get_shape()))
  return attention_reads


def attention_decoder(decoder_architecture,
                      decoder_state_initializer,
                      decoder_inputs,
                      decoder_input_lengths,
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

    decoder_input_lengths - list of integers with lengths of sentences

    attention_states: 3D Tensor [batch_size x attn_length x attn_size]

    output_size: Size of the output vectors; if None, we use cell.output_size

    num_heads: Number of attention heads that read from attention_states

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

  assert isinstance(final_encoder_states, (list)), "Final encoder states must be a list of lists"

  #TODO - introduce manning's attention mechanism that scores h_t * (W*h_s) instead of this version
  #THIS REQUIRES A REFACTOR

  #Bahdanu attention has more trainable weights (3 sets) compared to Manning's (1 set of weights)
  with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:

    #print("Reached the attention decoder. The type of the initial state is %s" % str(type(final_encoder_states)))

    #verify we have known shapes and nonzero inputs
    attn_length, attn_size = validate_attention_decoder_inputs(decoder_inputs, num_heads, attention_states)

    #store a scope reference as well as a datatype reference for book-keeping and debugging
    restore_scope = scope
    dtype = scope.dtype

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

    #we may want to initialize the attention mechanism
    if initial_state_attention:

      top_encoder_state_list = final_encoder_states[-1]
      reshaped_top_encoder_state = _prepare_top_lstm_encoder_state_for_attention(top_encoder_state_list) #makes one tensor if top encoder layer is bidirectional

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
      

      # If loop_function is set, we use it instead of real decoder_inputs.
      if loop_function is not None and previous_decoder_output is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          decoder_input = loop_function(previous_decoder_output, decoder_time_step)

      #TODO - with rank should probably be used throughout this file a lot more
      input_size = decoder_input.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % decoder_input.name)

      #attentions is a list, probably of length one because num_heads is probably one, but it is still a list.
      #decoder_input is not, it's a tensor, so we put it in a list.
      attentive_network_input = linear( [decoder_input] + attentions, input_size, True)

      #This is similar to the encoder mechanism we wrote earlier but the main difference
      # (other than the whole-attention-mechanism thing) is that this decoder call runs for ONE time step
      #decoder output is a list with a single tensor. eventually this list could be removed
      #decoder hidden states is a list of lists of tensors
      decoder_output, decoder_hidden_states = one_step_decoder(decoder_architecture,
                                                              attentive_network_input,
                                                              decoder_input_lengths,
                                                              decoder_hidden_states,
                                                              dtype=dtype)

      output_size = decoder_output[0].get_shape().with_rank(2)[1] #just for readability

      #TODO - this probably needs to become [-1][0] because this is a list of lists, so this assertion should catch it
      #print("I am looking for a list. List element zero is type %s" % str(hidden_states[-1][0].__class__.__name__))
      assert decoder_hidden_states[-1].__class__.__name__ == 'list', "Decoder hidden state's elements are lists."
      assert len(decoder_hidden_states[-1]) == 1, "decoder hidden state at top layer needs to only have one lstmstatetuple in the list. this is because we pass it directly as the query to the bahdanu attention mechanism, which expects an lstm tuple"
      
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

      #Now that we have all the pieces for the output from the decoder, we have one final parameter to train,
      # namely, the weight and bias vector that will be used to transform the dimensionality of our output
      # from the decoder into the dimensionality specified by the output projection
      with variable_scope.variable_scope("output_projection"):
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
                                decoder_input_lengths, #TODO - this needs to get factored in when/if _PAD is removed, so keep this around
                                final_encoder_states,
                                attention_states,
                                num_symbols,
                                embedding_size,
                                embedding_algorithm="network",
                                train_embeddings=True,
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

    #The embedding inputs represent the lookups in the embedding table for each symbol in the decoder, which we 
    # may or may not use, depending on if we are training or actively decoding. 
    decoder_embedding_inputs, embedding = embeddings.get_word_embeddings(decoder_inputs,
                                                                          num_symbols,
                                                                          embedding_size,
                                                                          embed_language="target",
                                                                          scope_name="decoder_embeddings",
                                                                          embed_algorithm=embedding_algorithm,
                                                                          train_embeddings=train_embeddings,
                                                                          return_list=True,
                                                                          dtype=dtype)

    #If we are actively decoding, then we don't care about the decoder inputs because we generate them
    # via a feed_previous option
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None

    #embedding_inputs = [ embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs ]

    return attention_decoder(
        decoder_architecture,
        decoder_state_initializer,
        decoder_embedding_inputs,
        decoder_input_lengths,
        final_encoder_states, #TODO this is a LIST, make changes accordingly
        attention_states,
        output_size=None,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)