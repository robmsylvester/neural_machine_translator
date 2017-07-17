from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model_utils
import vocabulary_utils

FLAGS = tf.app.flags.FLAGS


class seq2seqEDA(object):
  """Sequence-to-sequence attention model

    Creates an Encoder-Decoder architecture with attention mechanism, with support
    for arbitrary geometries for encoder and decoder lstm/gru stacks via a json
    file defining connections. In addition, a sampled softmax allows for really
    large vocabularies.

    For more info:
    http://arxiv.org/abs/1412.7449
    http://arxiv.org/abs/1409.0473
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               max_clipped_gradient,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               min_learning_rate,
               encoder_decoder_json_path, #TODO - this dude should probably just absorb the attention mechanism model parameters too
               max_encoder_length,
               max_decoder_length,
               softmax_sample_size=512,
               forward_only=False,
               dtype=tf.float32):
    """Create the model that can be called by the runner

    Args:
      source_vocab_size: size of the source vocabulary, usually defined by a flag.

      target_vocab_size: size of the target vocabulary, usually defined by a flag.

      max_gradient_norm: the maximum clipped gradient norm

      batch_size: the size of the batches used for training. when decoding, this defaults to one.

      learning_rate: learning rate to start with.

      learning_rate_decay_factor: decay learning rate by this much when needed.

      encoder_decoder_json_path: the file location of the json encoder / decoder architecture

      softmax_sample_size: number of samples for sampled softmax.

      forward_only: if set, we do not construct the backward pass in the model.

      dtype: the data type to use to store internal variables.

    """
    self.global_step = tf.Variable(0, trainable=False)
    self.dtype = dtype
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.batch_size = batch_size
    self.minimum_learning_rate = tf.constant(min_learning_rate, dtype=tf.float32)
    self.learning_rate_decay_factor = tf.constant(learning_rate_decay_factor, dtype=tf.float32)
    self.max_encoder_length = max_encoder_length
    self.max_decoder_length = max_decoder_length
    self.encoder_decoder_json_path = encoder_decoder_json_path

    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=True, dtype=self.dtype)

    self.learning_rate_decay_op = self.learning_rate.assign(
        tf.maximum(self.minimum_learning_rate, self.learning_rate * self.learning_rate_decay_factor, "learning_rate_decay"))

    #Load the JSON architecture stacks for LSTMs/GRUs and verify them
    self.encoder_architecture, self.decoder_architecture = model_utils.load_encoder_decoder_architecture_from_json(self.encoder_decoder_json_path, FLAGS.decoder_state_initializer)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None


    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if softmax_sample_size > 0 and softmax_sample_size < self.target_vocab_size:

      #The top decoder layer cannot be bidirectional, so we don't check for it right now.
      # If it is allowed later, this will need to have an additional check to double the input parameter to the create_output_projection function
      last_decoder_layer = next(reversed(self.decoder_architecture["layers"]))
      output_projection_input_size = self.decoder_architecture["layers"][last_decoder_layer]["hidden_size"]

      # This needs to actually be taken from the encoder_decoder_json architecture's top layer
      output_projection = model_utils._create_output_projection(self.target_vocab_size,
                                                                output_projection_input_size)

      weights = output_projection[0]
      biases = output_projection[1]
      weights_t = output_projection[2]


      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])

        # We need to compute the sampled_softmax_loss using 32bit floats to avoid numerical instabilities.
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=weights_t,
                biases=biases,
                labels=labels,
                inputs=tf.cast(logits, tf.float32),
                num_sampled=softmax_sample_size,
                num_classes=self.target_vocab_size),
                dtype)

      #Assign the previously declared function to be our loss function
      softmax_loss_function = sampled_loss


    # The seq2seq function: we use embedding for the input and attention.
    def sequence_to_sequence_function(encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths, do_decode):
      return model_utils.encoder_decoder_attention(
          encoder_inputs,
          decoder_inputs,
          encoder_input_lengths,
          decoder_input_lengths,
          self.encoder_architecture,
          self.decoder_architecture,
          decoder_state_initializer=FLAGS.decoder_state_initializer,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_algorithm=FLAGS.embedding_algorithm,
          num_heads=FLAGS.num_attention_heads,
          embedding_size=FLAGS.encoder_embedding_size,
          train_embeddings=FLAGS.train_embeddings or FLAGS.embedding_algorithm == 'network', #TODO, obviously fix this
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)



    # Feeds for inputs are lists of integers representing words
    self.encoder_inputs, self.decoder_inputs,\
     self.encoder_input_lengths,self.decoder_input_lengths, self.target_weights = self.create_encoder_decoder_input_placeholders()


    self.vocab_perplexities = tf.placeholder(tf.int32,
                                            shape=[target_vocab_size, FLAGS.vocab_boost_occurrence_memory],
                                            name="vocab_perplexities")

    self.target_weight_boosting_op = self.boost_decoder_weights(self.target_weights, self.vocab_perplexities)

    #We need to offset the _GO symbol addition by shifting our targets by one index to the right.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]


    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = model_utils.run_eda_architecture(
                                                self.encoder_inputs,
                                                self.decoder_inputs,
                                                self.max_encoder_length,
                                                self.max_decoder_length,
                                                self.encoder_input_lengths,
                                                self.decoder_input_lengths,
                                                targets,
                                                self.target_weights,
                                                lambda x, y, x_l, y_l : sequence_to_sequence_function(x, y, x_l, y_l, True),
                                                softmax_loss_function=softmax_loss_function)

      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        self.outputs = [tf.matmul(output, weights) + biases for output in self.outputs]


    else:
      self.outputs, self.losses = model_utils.run_eda_architecture(
                                                self.encoder_inputs,
                                                self.decoder_inputs,
                                                self.max_encoder_length,
                                                self.max_decoder_length,
                                                self.encoder_input_lengths,
                                                self.decoder_input_lengths,
                                                targets,
                                                self.target_weights,
                                                lambda x, y, x_l, y_l : sequence_to_sequence_function(x, y, x_l, y_l, False), #we will pass in the arguments ourselves in the model_without_buckets function
                                                softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    #If we are doing backprop, we split the minimize() call into two calls, namely, gradients() and apply_gradients().
    #This allows us to play with the gradients before applying them
    if not forward_only:

      #Define an optimizer
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)

      #Grab the gradients from tensorflow form the list of trainable variables. 
      gradients = tf.gradients(self.losses, params)

      #Clip the gradients to avoid exploding gradients
      clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                       max_clipped_gradient)

      #Apply the gradients to the graph updates
      self.gradient_norms = norm
      self.updates = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

    #Save everything so we can reload after the power goes out because kitty unplugs the computer looking for his stupid mouse toy.
    self.saver = tf.train.Saver(tf.global_variables())

  def step(self,
          session,
          encoder_inputs,
          decoder_inputs,
          encoder_input_lengths,
          decoder_input_lengths,
          target_weights,
          forward_only=False):
    """Run a step of the model with the encoder and decoder inputs passed from either
      A. The training minibatch
      B. The validation set in its entirety
      C. A single example while actively decoding, in which case there is just one sentence in the "batch"

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      encoder_input_lengths: list of ints that store the length of the encoder language sentences excluding pad symbols
      decoder_input_lengths: list of ints that store the length of the decoder language sentences excluding pad and go symbols
      target_weights: list of numpy float vectors to feed as target weights. this is how much we care about the individual symbols
       about getting this particular token correct. naively, these are all 1, except padded words, which are 0
       this will be deprecated in favor of a dynamic rnn. in the event that certain words are boosted if we keep getting them
       wrong, then this target_weights will need to be adjusted
      forward_only: whether to do the backward step or only forward. in other words, does this do backpropagation or not.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees withmax encoder/decoder length
    """
    # Check if the sizes match.
    if len(encoder_inputs) != self.max_encoder_length:
      raise ValueError("Encoder length must be equal to the max encoder length,"
                       " %d != %d." % (len(encoder_inputs), self.max_encoder_length))
    if len(decoder_inputs) != self.max_decoder_length:
      raise ValueError("Decoder length must be equal to the max decoder length,"
                       " %d != %d." % (len(decoder_inputs), self.max_decoder_length))
    if len(target_weights) != self.max_decoder_length:
      raise ValueError("Weights length must be equal to the max decoder length,"
                       " %d != %d." % (len(target_weights), self.max_decoder_length))

    #Now, we build an input feed. This means creating names for variables
    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(self.max_encoder_length):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(self.max_decoder_length):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[self.max_decoder_length].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # We will also toss in the encoder and decoder lengths
    input_feed[self.encoder_input_lengths.name] = encoder_input_lengths
    input_feed[self.decoder_input_lengths.name] = decoder_input_lengths

    # Output feed: depends on whether we do a backward step or not.

    #Doing backpropagation
    if not forward_only:
      output_feed = [self.updates,  # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.losses]  # Loss for this batch.

    #Not doing backpropagation
    else:
      output_feed = [self.losses]  # Loss for this batch.
      for l in xrange(self.max_decoder_length):  # Output logits.
        output_feed.append(self.outputs[l])

    #Actually runs the network
    outputs = session.run(output_feed, input_feed)

    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs per time step.



  def get_batch(self, data, load_from_memory=True, use_all_rows=False, source_path=None, target_path=None, max_size=None):
    """Get a random batch of data, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      load_from_memory : boolean - if true, loads the dataset from memory by
        directly examining the buckets and randomly choosing from them. if false,
        will randomly choose lines fitting the sizing parameters of the bucket_id
        that was passed. notice that this will be slower.
      use_all_rows - ignore batch size and use every training sentence pair in data

      #TODO - finish implementing getting a batch from not the file

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    if load_from_memory:
      return self.get_batch_from_memory(data, use_all_rows=use_all_rows)
    else:
      raise NotImplementedError("Need to write get back from file method. use the memory flag now")
      #return self.get_batch_from_file(data, use_all_rows=use_all_rows, source_path=None, target_path=None, max_size=None)


  #Create placeholder variables for the encoder/decoder inputs
  def create_encoder_decoder_input_placeholders(self):

    # Feeds for inputs.
    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for i in xrange(self.max_encoder_length):
      encoder_inputs.append(tf.placeholder(tf.int32,
                                            shape=[None], #unknown batch_size
                                            name="encoder{0}".format(i)))
      
    for i in xrange(self.max_decoder_length + 1): #plus 1 because we added the Go symbol
      decoder_inputs.append(tf.placeholder(tf.int32,
                                            shape=[None],
                                            name="decoder{0}".format(i)))
      
      target_weights.append(tf.placeholder(self.dtype,
                                            shape=[None],
                                            name="weight{0}".format(i)))

    #now we store the lengths for the inputs
    encoder_input_lengths = tf.placeholder(tf.int32, shape=[None], name="encoderInputLengths")
    decoder_input_lengths = tf.placeholder(tf.int32, shape=[None], name="decoderInputLengths")

    return encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths, target_weights

  #TODO - write this, and write this within the attention decoder at teach step
  def boost_decoder_weights(self, weights, perplexity_matrix):
    weights = weights
    perplexity_matrix=perplexity_matrix
    #Yeah, this function does nothing



  def prepare_encoder_and_decoder_inputs(self, data, size=None):
    """Cleaning encoder and decoder inputs amounts to padding the inputs to the size of the max sentence for
    both the encoder and decoder, as well as adding a go symbol to the beginning of the decoder sentence.
    This function will also reverse the input encoder sentence. """

    #Args:
    #self - the seq2seq class
    #data - sentence integer training pairs as a list of tuples
    #size - int, the number of data rows to read
    encoder_inputs = []
    decoder_inputs = []
    encoder_input_lengths = [] #we'll need this to pass to the dynamic rnn's as their sequence length arguments
    decoder_input_lengths = [] #we'll need this to pass to the dynamic rnn's as their sequence length arguments

    # Get a random batch of encoder and decoder inputs from data
    for _ in xrange(size):
      encoder_input, decoder_input = random.choice(data)

      # Encoder inputs are padded temporarily
      encoder_pad = [vocabulary_utils.PAD_ID] * (self.max_encoder_length - len(encoder_input))

      # Then we reverse the inputs as per Cho's recommendations
      #we because we can't have sequence lengths on the padded input
      #encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      #not reversing inputs. using dynamic length setting instead
      encoder_inputs.append(encoder_input+encoder_pad)
      #encoder_inputs.append(list(reversed(encoder_pad + encoder_input)))

      # Decoder inputs get an extra "GO" symbol since we feed previous outputs to a decoder, and this will handle that placeholder spot
      decoder_pad_size = self.max_decoder_length - len(decoder_input) - 1

      # We pad the decoder inputs to fill the rest of the sequence placeholder as well
      decoder_inputs.append([vocabulary_utils.GO_ID] + decoder_input +
                            [vocabulary_utils.PAD_ID] * decoder_pad_size)

      assert len(encoder_input)
      assert len(decoder_inputs[_]) == self.max_decoder_length, "Incorrect. Looking at\n\n%s\n\npad size is %d, decoder length is %d" % (str(decoder_inputs[_]), decoder_pad_size, len(decoder_input))

      # And we store the sequence length
      encoder_input_lengths.append(len(encoder_input))
      decoder_input_lengths.append(len(decoder_input) + 1) #TODO - run some tests on this +1 and make sure we aren't off by one here because of _EOS

    return encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths


  def boosted_weight_target_symbols(self, decoder_inputs, decoder_size):
    pass


  def default_weight_target_symbols(self, decoder_inputs, decoder_size):

    #decoder size - the max sentence decoder length
    #decoder inputs - the batch size length array inputs for the decoder at that time step

    """As it stands now in this implementation, we don't really care about learning the padding term
    at all, so we zero it out in the loss function calculation. We also zero out the final symbol of
    the decoder as well, since it gets a + Every other target gets an equal weight of
    1.0. These weights will be multiplied by their corresponding decoder indexes in the loss function
    when the overall softmax cross-entropy is calculated. In time, it might be worthwhile to dynamically
    adjust the target weights if we find that we are misclassifying certain labels. One way to do this would
    be to examine the perplexity of a sentence and boost the weights of those particular labels based
    on how well we do on that sentence.

    For example, if a sentence "I have a dog" translated to "Io non ho un gatto" (I don't have a cat), then
    perhaps we need to boost the importance of getting the word dog". If we nailed
    the sentence, and it therefore has low perplexity, the weights for these logits could decrease.
    This is just an idea, and probably just a repetition of what the other layers are learning.

    For now, we simply treat PAD_ID as a weight of 0, and we treat the final length index as 0. all other
    words get an importance of 1."""

    default_weights = []

    for length_idx in xrange(decoder_size):
      batch_weight = np.ones( shape=len(decoder_inputs), dtype=np.float32)
      for batch_idx in xrange(len(decoder_inputs)):
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == vocabulary_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      default_weights.append(batch_weight)
    return default_weights


  def get_batch_from_memory(self, data, use_all_rows=False):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: list of tuple pairs of input and output data that we use to create a batch.
      load_from_memory : boolean - if true, loads the dataset from memory by
        directly examining the buckets and randomly choosing from them. if false,
        will randomly choose lines fitting the sizing parameters
        that was passed. notice that this will be slower.
      use_all_rows : boolean - if true, ignores batch size and loads entire dataset in "data"

    Returns:
      The quintuple (encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size = self.max_encoder_length
    decoder_size = self.max_decoder_length
    encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths = self.prepare_encoder_and_decoder_inputs(data, size=self.batch_size if not use_all_rows else len(data))

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs = []
    batch_decoder_inputs = []
    batch_weights = []

    #fetch a coefficient that corresponds to how much we care about getting each decoder input correct
    #for now, it turns out we naively care about all targets the same, except _PAD, which we totally dont care about
    #this function takes care of the transpose for us as well.
    batch_weights_old = self.default_weight_target_symbols(decoder_inputs, decoder_size)

    #our inputs need to be reshaped so that we feed in the symbol at time step t for each example in the batch
    #encoder inputs are a list, so make an array out of the list and take the tranpose of the matrix to do this.
    #Shapes will now be (encoder_max_length, batch_size) and (decoder_max_length, batch_size) respectively.
    encoder_input_as_array = np.transpose(np.stack(encoder_inputs, axis=0)).astype(np.int32)
    decoder_input_as_array = np.transpose(np.stack(decoder_inputs, axis=0)).astype(np.int32)

    #split the array back into a list of symbols, which creates a list of [1,batch_size] shapes of length encoder_max length.
    #so we squeeze them into [batch_size,]
    #in the event we have a batch size of 1 (decoding, ie) this squeezes [1,1] to [] instead of [1], so we pass axis=0
    batch_encoder_inputs = [np.squeeze(i, axis=0) for i in np.split(encoder_input_as_array, encoder_size, axis=0)]
    batch_decoder_inputs = [np.squeeze(i, axis=0) for i in np.split(decoder_input_as_array, decoder_size, axis=0)]

    #decoder_input_copy = np.copy(decoder_inputs)
    #decoder_input_copy = np.transpose(decoder_input_copy, perm=[1,0,2])

    return batch_encoder_inputs, batch_decoder_inputs, encoder_input_lengths, decoder_input_lengths, batch_weights_old


  def get_batch_from_file(self, data, source_path, target_path, max_size):
    raise NotImplementedError
