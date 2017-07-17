import tensorflow as tf
import numpy as np
import vocabulary_utils
from tensorflow.python import shape
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.platform import gfile

FLAGS = tf.app.flags.FLAGS



#=================================================================
#
#	Embeddings.py
#
#	There are four types of embeddings that can be used in a network that are supported.
#
#	1) Network embeddings - This means that the first layer of the neural network is a set of
#                           weights that transforms the input shape into a defined embedding
#							size. We effectively ignore the unsupervised learning operations
#                           like glove, word2vec, or fasttext. This means IMPORTANTLY that the
#                           embedding parameters are trained by backprop.
#
#
#	2) Glove - 				The neural network first layer is the first layer of the encoder. Fed
#                           to this are the results of running the Glove algorithm on the vocabulary
#                           inputs, effectively just consulting the lookup table for each token and
#							getting its vector representation
#

#if FLAGS.embedding_type == "glove":
#	W = tf.constant(embedding, name="glove_trained_weight_embeddings")



def _determine_embedding_and_vocabulary_file(embed_language, embed_algorithm):
  if embed_language is None:
    assert embed_algorithm == "network", "If there is no passed embedding language, you must not pass None to embed_algorithm because it expects to use a pretrained embedding file to initialize tensors."
    return None, None
  
  elif embed_language == "source":
    vocab_file = FLAGS.data_dir + "/vocabulary_" + str(FLAGS.from_vocab_size) + ".from"
    if embed_algorithm == 'glove':
      embed_file = FLAGS.glove_encoder_embedding_file
  
  elif embed_language == "target":
    vocab_file = FLAGS.data_dir + "/vocabulary_" + str(FLAGS.to_vocab_size) + ".to"
    if embed_algorithm == 'glove':
      embed_file = FLAGS.glove_decoder_embedding_file
  else:
    raise ValueError("Embed language must be None, source or target")
  return embed_file, vocab_file


#TODO - refactor this so it doesn't have repeated code for all four cases
def get_word_embeddings(inputs,
                        num_symbols,
                        embed_size,
                        embed_language,
                        scope_name="embeddings",
                        embed_algorithm="network",
                        train_embeddings=True,
                        return_list=True,
                        dtype=None):

  if embed_algorithm != "network":
    print("determining embedding file. embed language is %s" % embed_language)
    embed_file, vocab_file = _determine_embedding_and_vocabulary_file(embed_language, embed_algorithm)

  #No unsupervised learning algorithm
  if embed_algorithm == "network":
    print("***Note - No unsupervised embedding algorithm is present. Word embeddings will be randomized and trained by backpropagation.")
    #create embeddings - this will eventually be moved elsewhere when the embeddings are no longer trained

    with variable_scope.variable_scope(scope_name) as scope:
      print("using network embeddings with size %d" % embed_size)
      emb = tf.get_variable(scope_name + "_network_embeddings",
                            shape = [num_symbols, embed_size],
                            initializer = tf.random_uniform_initializer(-1.0, 1.0),
                            trainable = True, #have to be trainable because embed algorithm is none.
                            dtype = dtype)

      #get the embedded inputs from the lookup table
      #these will be trained by backpropagation
      embedded_inputs = tf.nn.embedding_lookup(emb, inputs)

      #these embedded inputs came in as a list. now they will be of the shape (bucket_size, batch_size, embed_size)
      #if return list is true (static encoder api), so we reshape them back into a list of length bucket_size containing tensors of shape (batch_size, embed_size)
      if return_list:
        return tf.unstack(embedded_inputs), emb
      else: 
        return embedded_inputs, emb

  elif embed_algorithm == 'glove':
    print("\tWord embeddings will be initialized by glove")

    if train_embeddings:
      print("\tWord embeddings will still be trained by backprop")
    else:
      print("\tWord embeddings will NOT be trained by backprop")

    with variable_scope.variable_scope(scope_name) as scope:
      emb = tf.get_variable(scope_name+"_glove_embeddings",
                            initializer=vocabulary_utils.initialize_glove_embeddings_tensor(num_symbols, embed_size, embed_file, vocab_file, dtype=dtype),
                            trainable=train_embeddings,
                            dtype=dtype)

      #get the embedded inputs from the lookup table
      #these will be trained by backpropagation only if train_embeddings is enabled.
      embedded_inputs = tf.nn.embedding_lookup(emb, inputs)

      #these embedded inputs came in as a list. now they will be of the shape (max_time, batch_size, embed_size),
      #but if they are being used in a static rnn we need a list of length max_time with tensors being of shape (batch_size, embed_size)
      if return_list:
        return tf.unstack(embedded_inputs), emb
      else: 
        return embedded_inputs, emb
  else:
    raise NotImplementedError