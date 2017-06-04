import tensorflow as tf
import custom_core_rnn as core_rnn
from tensorflow.python import shape
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
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

#TODO - this needs to become dynamic instead of static rnn's, probably. depending on timer calls. fuckin' padding.
def run_encoder_NEW(encoder_json,
                    encoder_inputs,
                    num_encoder_symbols, #TODO - needs to be keyword arg because this may be coming from unsupervised embedding thats not in network
                    embedding_size, #TODO - also needs to be keyword arg because this may be coming from unsupervised embedding thats not in network
                    nematus_state_values=False, #TODO
                    dtype=None):

  # embeddings need to become non-updated-by-backprop embeddings from unsupervised glove, word2vec, or fasttext options
  with variable_scope.variable_scope("encoder", dtype=dtype) as scope:
    dtype=scope.dtype

    #print("encoder inputs right now have length of %d and each has shape %s" % (len(encoder_inputs), str(encoder_inputs[0].get_shape())))

    #create embeddings - this will eventually be moved elsewhere when the embeddings are no longer trained
    with variable_scope.variable_scope("embeddings") as scope:
      embeddings = tf.get_variable("encoder_embeddings",
                                  shape=[num_encoder_symbols, embedding_size],
                                  initializer=tf.random_uniform_initializer(-1.0, 1.0),
                                  dtype=dtype)

      #get the embedded inputs from the lookup table
      embedded_encoder_inputs = tf.nn.embedding_lookup(embeddings, encoder_inputs)

      #these embedded inputs came in as a list so they will be of the shape (bucket_size, batch_size, embed_size), but we need
      # them to be a sequence input for the lstm layers, so we reshape them back into a list of length bucket_size containing tensors of shape (batch_size, embed_size)
      embedded_encoder_inputs = tf.unstack(embedded_encoder_inputs)

    #print("the embedded encoder inputs each have shape %s, and there are %d of them in the list" % (str(embedded_encoder_inputs[0].get_shape()), len(embedded_encoder_inputs)))

    current_layer = 0
    cell_outputs = OrderedDict() #indexed by layer name in json
    cell_states = OrderedDict() 

    #TODO - this NEEDS to become dynamic rnn
    for layer_name, layer_parameters in encoder_json["layers"].iteritems(): #this is an ordered dict
      #print("encoder is analyzing layer %s" % layer_name)

      with variable_scope.variable_scope(layer_name) as scope:
        
        #TODO, take this and the loop and refactor into a get_inputs function
        #first loop will get embeddings, later loops will use previous iteration outputs, plus any residual connections
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

        #print("layer %s has %d total inputs in the input list." % (layer_name, len(input_list)))

        #combine these residual inputs following this layer's merge mode
        if layer_parameters['input_merge_mode'] == 'concat':
          #inputs = tf.cond(len(input_list) > 1,
          #  lambda:tf.concat(input_list, axis=1),
          #  lambda:input_list[0])
          #print("the shape of the input list BEFORE concatenating is shape %s and there are %d of them" % (str(inputs[0].get_shape()), len(input_list)))
          inputs = tf.unstack(tf.concat(input_list, axis=2)) #we unstack to get a list
          #print("the shape of the inputs AFTER concatenating is %s and there are %d of them" % (str(inputs[0].get_shape()), len(inputs)))
        elif layer_parameters['input_merge_mode'] == 'sum':
          #inputs = tf.unstack(tf.add_n([i for i in input_list])) #TODO - correct here? unstack check
          #inputs = tf.cond(len(input_list) > 1,
          #  lambda:tf.unstack(tf.add_n(input_list)),
          #  lambda:input_list[0])
          #print("******************************************")
          inputs = tf.unstack(tf.add_n(input_list))
          #print("going to sum inputs into layer %s\nThe dimensionality of the first tensor in the input list is %s" % (layer_name, str(inputs[0].get_shape())))
          #print("the shape of the inputs after summing is %s and there are %d of them" % (str(inputs[0].get_shape()), len(inputs)))
        else:
          assert current_layer==0, "The input merge mode must be concat or sum for all other layers but 0" #remove this text
          assert len(input_list)==0, "The current layer should be layer 0 if there are no inputs from other layers" #remove this test
          inputs = embedded_encoder_inputs


        #print("About to run the network layer. Inputs have shape %s and length %d" % (str(inputs[0].get_shape()), len(inputs)))
        #create/get the cells, and run them
        #this will give us the outputs, and we can combine them as necessary
        #c stands for cell, out for outputs, f for forward, b for backward
        if layer_parameters['bidirectional']:
          cf = _create_encoder_cell(layer_parameters)
          cb = _create_encoder_cell(layer_parameters)
          out_fb, out_f, out_b, state_f, state_b = core_rnn.static_bidirectional_rnn(cf, cb, inputs, dtype=dtype)
          #print("bidirectional rnn ran.\n\ttype of out_fb is %s\n\ttype of out_f is %s\n\ttype of out_b is %s\n\ttype of state_f is %s\n\ttype of state_b is %s" % (str(type(out_fb)), str(type(out_f)), str(type(out_b)), str(type(state_f)), str(type(state_b))))
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
          cell_states[layer_name] = [state_f,state_b]

        else:
          cf = _create_encoder_cell(layer_parameters)

          #out_f is a list of tensor outputs, state_f is an LSTMStateTuple or GRU State, so we put the state in a single-element list so that both return lists.
          out_f, state_f = core_rnn.static_rnn(cf, inputs, dtype=dtype)
          #print("unidirectional rnn ran.\n\ttype of out_f is %s\n\ttype of state_f is %s" % (str(type(out_f)),str(type(state_f))))
          cell_outputs[layer_name] = [out_f]
          cell_states[layer_name] = [state_f]

      top_layer = layer_name # just for readability
      current_layer += 1

    #end main loop
    #we do care about the states, but only the top most state, which is what is stored in state_f unless it is bidirectional
    #state tuples are now stored as (cell_state, hidden_state)
    #if len(stack_state) == 2:
    #  assert type(stack_state[0]) is core_rnn_cell_impl.LSTMStateTuple, "top-level forward LSTM returned state should be an LSTMStateTuple. state is tuple is now true by default in TF."
    #  assert type(stack_state[1]) is core_rnn_cell_impl.LSTMStateTuple, "top-level backward LSTM returned state should be an LSTMStateTuple. state is tuple is now true by default in TF."
    #  stack_state = tf.concat(stack_state, axis=1) #concatenate the bidirectional states. #TODO - make sure this axis is right
    #elif len(stack_state) == 1:
    #  assert type(stack_state[0]) is core_rnn_cell_impl.LSTMStateTuple, "top-level forward LSTM returned state should be an LSTMStateTuple. state is tuple is now true by default in TF."
    #  stack_state = stack_state[0]

    #we do care about the outputs, but they might be bidirectional outputs too, so we must check that as well
    #for now, we will just concatenate these by default

    #stack_output = tf.cond( len(cell_outputs[layer_name]) == 2,
    #  lambda: tf.concat(cell_outputs[layer_name], axis=1),
    #  lambda: cell_outputs[layer_name][0])

    if FLAGS.decoder_state_initializer == 'nematus':
      raise NotImplementedError, "Haven't written the nematus decoder state initializer yet, and currently this function only returns and tracks top-level states in the loop."

    #We do this anyway outside of the if statement, because the concatenation won't do anything otherwise, and the unstack 
    if len(cell_outputs[top_layer]) > 1:
      print("WARNING - your top layer cell outputs more than a single tensor at each time step. Perhaps it is bidirectional with no output merge mode specified. These tensors will be concatenated along axis 1. You should change this in the JSON to be 'concat' for readability, or 'sum' if you want the tensors element-wise added.")
    
    stack_output = tf.unstack(tf.concat(cell_outputs[top_layer], axis=1))
    stack_states = [ cell_states[l] for l in cell_states ] # list of lists of states

    #print("length of stack states is %d" % len(stack_states))
    #print("top stack state is type %s" % str(type(stack_states[-1])))

    #if len(cell_states[layer_name]) == 2:
    #  stack_states = [tf.concat(cell_states[layer_name], axis=1)] #returns a list of length 1
    #elif len(cell_states[layer_name]) == 1:
    #  stack_states = cell_states[layer_name][0][0] #returns a list of length 1
    #else:
    #  raise ValueError, "Expected 1 or 2 total output tensors in list of top layer of encoder stack states"

    #This output should be a list with one element
    return stack_output, stack_states


# Concatenation of encoder outputs to put attention on.
# TODO - encoder hidden size here is written here instead of output_size. NEED TO PASS NEW PARAMETER BECAUSE TOP LAYER MIGHT BE BIDIRECTIONAL
# done this way because it is assumed top layer is not bidirectional, or otherwise
# have any reason to use a different variable.
def get_attention_state_from_encoder_outputs(encoder_outputs,
                                            scope=None,
                                            dtype=None):
  with variable_scope.variable_scope(scope or "attention_from_encoder_outputs", dtype=dtype) as scope:
    assert isinstance(encoder_outputs, (list)), "Encoder outputs must be a python list. Instead it is of type %s" % str(type(encoder_outputs))
    #print("Length of encoder outputs is %d" % len(encoder_outputs))
    dtype=scope.dtype
    #print("get attention state from encoder outputs has been called.")

    #THIS IS BUGGY, because top layer might be bidirectional
    top_states = [array_ops.reshape(enc_out, [-1, 1, FLAGS.encoder_hidden_size]) for enc_out in encoder_outputs]
    #print("top states has a total of %d tensors" % len(top_states))
    #print("the first of these tensors has shape %s" % str(top_states[0].get_shape()))
    attention_states = array_ops.concat(top_states, 1) #does this need to be axis=2
    #print("after concatenating, the attention states are shape %s" % str(attention_states.get_shape()))
    return attention_states 




"""

def run_encoder_DEPRECATED_DELETE_THIS(encoder_inputs,
            num_encoder_symbols,
            embedding_size,
            dtype=None):

  print("length of encoder inputs is %d" % len(encoder_inputs))
  print("shape of encoder input 0 is %s" % str(encoder_inputs[0].get_shape()))

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
    print("encoder is returning encoder outputs which has len %d " % len(encoder_outputs))
    print("encoder output tensors have shape %s " % str(encoder_outputs[0].get_shape()))
    print("encoder state cell state tensor has shape %s " % str(encoder_state[0].get_shape()))
    print("encoder state hidden state tensor has shape %s " % str(encoder_state[1].get_shape()))
    assert type(encoder_state) is core_rnn_cell_impl.LSTMStateTuple, "encoder_state should be an LSTMStateTuple. state is tuple is now true by default in TF."
    assert len(encoder_outputs) == len(encoder_inputs), "encoder input length (%d) should equal encoder output length(%d)" % (len(encoder_inputs), len(encoder_outputs))
    return encoder_outputs, encoder_state
"""