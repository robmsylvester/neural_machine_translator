from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python import shape 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
#from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops #TODO - make this not so fucking old
from tensorflow.python.ops import nn_ops #TODO - make this not so fucking old
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

#from tensorflow.contrib.rnn.python.ops import core_rnn
import custom_core_rnn as core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl

#This is very similar to a dot product in this implementation between inputs and weights
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access

FLAGS = tf.app.flags.FLAGS

def _create_decoder_lstm(hidden_size, use_peepholes, init_forget_bias, dropout_keep_prob):
  c = core_rnn_cell_impl.LSTMCell(hidden_size, #number of units in the LSTM
                use_peepholes=use_peepholes,
                initializer=tf.contrib.layers.xavier_initializer(),
                forget_bias=init_forget_bias)
  if dropout_keep_prob < 1.0:
    c = core_rnn_cell_impl.DropoutWrapper(c, output_keep_prob=dropout_keep_prob)
  return c



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
      with variable_scope.variable_scope(scope or "output_proj", dtype=dtype) as scope:
        prev = tf.add(tf.matmul(prev, output_projection[0]), output_projection[1])

    prev_symbol = tf.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


def validate_attention_decoder_inputs(decoder_inputs, num_heads, attention_states):

  if not decoder_inputs:
    raise ValueError("Your attention decoder has no decoder inputs")
  if num_heads < 1:
    raise ValueError("The number of heads to the attention decoder must be a positive integer. It is %d" % num_heads)
  if attention_states.get_shape()[2].value is None:
    raise ValueError("The attention_size of the attention states must be known. Attention states come in the order (batch_size, attention_length, attention_size). This shape of this input now is %s" %
                     attention_states.get_shape())
  
  attn_length = attention_states.get_shape()[1].value
  if attn_length is None:
    attn_length = array_ops.shape(attention_states)[1]

  attn_size = attention_states.get_shape()[2].value

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
def decoder_rnn(attn_input, hidden_states, num_layers=None, scope=None):

  #TODO - this
  if scope is None:
    raise ValueError("scope cannot be None in decoder_rnn. This has not been implemented. Easy fix, so do it Rob.")

  if num_layers != len(hidden_states):
    raise ValueError("expected %d hidden states. instead only see %d hidden states" % (num_layers, len(hidden_states)))

  with variable_scope.variable_scope(scope.name+'/fw0') as scope:
    fw0 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)
    outputs0, hidden_states[0] = fw0(attn_input, hidden_states[0], scope=scope)
  
  with variable_scope.variable_scope(scope.name+"/fw1") as scope:
    fw1 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)
    outputs1, hidden_states[1] = fw1(outputs0, hidden_states[1], scope=scope)
  
  with variable_scope.variable_scope(scope.name+"/fw2") as scope:
    fw2 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)

    #add a residual connection to the outputs from the first layer
    #we don't need to call unstack here because the inputs into the __call__ function of the LSTMCell
    #are a single tensor, not a list of tensors.
    inputs2 = tf.add_n([outputs0,outputs1], name="residual_decoder_layer2_input")
    outputs2, hidden_states[2] = fw2(inputs2, hidden_states[2], scope=scope)
  
  with variable_scope.variable_scope(scope.name+"/fw3") as scope:
    fw3 = _create_decoder_lstm(FLAGS.decoder_hidden_size,
                        FLAGS.decoder_use_peepholes,
                        FLAGS.decoder_init_forget_bias,
                        FLAGS.decoder_dropout_keep_probability)

    #add a reisdual connection to the outputs from the second layer
    inputs3 = tf.add_n([outputs1, outputs2], name="residual_decoder_layer4_input")
    outputs3, hidden_states[3] = fw3(inputs3, hidden_states[3], scope=scope)

  output_size = fw3.output_size

  #we only care about the final output, but we need all the hidden cell states to pass back to this function later
  return outputs3, hidden_states, output_size


"""
def attention_mechanism(input_vector, dec_state ):

  #The input_vector is the input state to attention mechanism.


  attention_reads = []

  #if the query is a 
  if nest.is_sequence(query_state):
    query_










      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):

          y = linear(query, attention_vec_size, True)

          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds
"""



def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
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


  with variable_scope.variable_scope(scope or "attention_decoder", dtype=dtype) as scope:

    #verify we have known shapes and nonzero inputs
    attn_length, attn_size = validate_attention_decoder_inputs(decoder_inputs, num_heads, attention_states)

    #store a scope reference as well as a datatype reference for book-keeping and debugging
    restore_scope = scope
    dtype = scope.dtype

    #we need to store the batch size of the decoder inputs to use later for reshaping.
    # because these come in as a list of tensors, just take the first one.
    #batch_size = tf.get_shape(decoder_inputs[0]).as_list()[0]

    #assert batch_size == array_ops.shape(decoder_inputs[0])[0], "expected equality. didn't get it"
    
    #batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.

    #attn_length = attention_states.get_shape()[1].value


    #if attn_length is None:
      #attn_length = array_ops.shape(attention_states)[1]
    #attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.

    print("the attention decoder has been called with attention states sized " + str(attention_states.get_shape()))

    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])

    print("the attention decoder has reshaped the attention state to size " + str(hidden.get_shape()))

    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
      print("the attention decoder has reshaped the attention state to a new size using 1x1 convolution. now size is " + str(hidden_features[a].get_shape()))

    outputs = []
    prev = None
    num_decoder_layers = 4
    hidden_states = [initial_state for _ in xrange(num_decoder_layers)]



    def attention(query_state):
      """Put attention masks on hidden using hidden_features and query."""
      #query state is the last decoder state from the top layer of the LSTM decoder stack that we can pass to the attention model
      # after each decoder step

      attention_reads = [] #This will be built dynamically as we read through the attention heads

      assert query_state.__class__.__name__ == 'LSTMStateTuple', "The decoder state passed to the attention model must be an LSTMStateTuple (c,h)"

      #TODO - probably can clean this up
      #first, flatten the cell state and hidden state
      query_list = nest.flatten(query_state)
      for q in query_list:  # Check that ndims == 2 if specified.
        ndims = q.get_shape().ndims
        if ndims:
          assert ndims == 2

      query_state = array_ops.concat(query_list, 1)
      print("The attention mechanism has flattened the query shape to " + str(query_state.get_shape()))
      
      for head_idx in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):

          print("applying linear transformation with attention vector size " + str(attention_vec_size))

          #we apply a linear transformation to the (batch_size, decoder_hidden_size*2 query state to transform it into the size of our attention vector)
          #this i
          y = linear(query_state, attention_vec_size, True)

          print("after transformation, new shape is " + str(y.get_shape()))

          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])

          print("after reshape squeeze, new shape is " + str(y.get_shape()))
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[head_idx] + y),
                                  [2, 3])

          print("after attention hyperbolic tangent function, new shape is " + str(s.get_shape()))
          a = nn_ops.softmax(s)

          print("after softmax function, new shape is " + str(a.get_shape()))

          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])

          print("after attention weighted representation of vector, new shape is " + str(d.get_shape()))
          attention_reads.append(array_ops.reshape(d, [-1, attn_size]))
          print("this representation has been reshaped to the final shape of " + str(ds[-1].get_shape()))
      return attention_reads


    #we need to store the batch size of the decoder inputs to use later for reshaping.
    # because these come in as a list of tensors, just take the first one.
    # this will store a TENSOR of one-dimension with the batch_size shape.
    batch_size = tf.shape(decoder_inputs[0])[0]

    #the size of the batch attention will be batch_size x attention size, and we stack these values along the 0 axis
    batch_attn_size = tf.stack([batch_size, attn_size])
    
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
    ]

    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)

    for i, inp in enumerate(decoder_inputs):

      print("Running attention mechanism on decoder input index %d" % i)

      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)

      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      attentive_input = linear([inp] + attns, input_size, True)

      #initialize the hidden states of the network if i==0

      #Run the RNN
      decoder_output, hidden_states, output_size = decoder_rnn(attentive_input,
                                                              hidden_states,
                                                              num_layers=num_decoder_layers,
                                                              scope=restore_scope)
      decoder_state = hidden_states[-1]
      scope = restore_scope

      #print("The shape of the final decoder state that will be passed to the attention mechanism is" + str(decoder_state.get_shape()))

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(decoder_state)
      else:
        attns = attention(decoder_state)

      print("after calling attention mechanism, the output shape of the first item in the attns list is " + str(attns[0].get_shape()))

      with variable_scope.variable_scope("AttnOutputProjection"):
        decoder_output = linear([decoder_output] + attns, output_size, True)
      if loop_function is not None:
        prev = decoder_output
      outputs.append(decoder_output)

  return outputs, decoder_state





def embedding_attention_decoder(decoder_inputs,
                                initial_state,
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
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
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

  #print(type(output_projection[0]))
  #print(type(output_projection[1]))
  #print("var scope of w_t from custom contrib is " + str(output_projection[0].get_variable_scope().name))

  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
    ]
    return attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        output_size=None,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)