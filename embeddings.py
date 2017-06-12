#TODO - remove unneeded dependencies

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
from tensorflow.python import shape 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops #TODO - make this not so fucking old
from tensorflow.python.ops import nn_ops #TODO - make this not so fucking old
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import custom_core_rnn as core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.platform import gfile

#This is very similar to a dot product in this implementation between inputs and weights
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access
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
#
#
#
#
#
#
#
#
#

#if FLAGS.embedding_type == "glove":
#	W = tf.constant(embedding, name="glove_trained_weight_embeddings")


#TODO - abstract away this embedding file flag into a full argument passed earlier on
def initialize_glove_embeddings_tensor(num_enc_symbols, embed_size, embedding_file=None, dtype=None):
  if embedding_file is None:
  	embedding_file = FLAGS.glove_embedding_file

  if gfile.Exists(embedding_file):
  	embeddings = np.zeros(shape=(num_enc_symbols, embed_size), dtype=np.float32) #TODO - fix this dtype

  	#PAD can remain all zeros. Going away anyways

  	#randomize the _GO symbols
  	embeddings[1] = np.random.uniform(low=-1.0, high=1.0, size=embed_size)

  	#randomize the _EOS symbols
  	embeddings[2] = np.random.uniform(low=-1.0, high=1.0, size=embed_size)

  	#randomize the _UNK symbols
  	embeddings[3] = np.random.uniform(low=-1.0, high=1.0, size=embed_size)

  	#load the pre-trained vectors
  	num_special_symbols = 4
  	lines_read = 0
  	limit = 20

  	with gfile.GFile(embedding_file, mode="rb") as f:
  		for line in f:
  			vector = line.split()[1:]
  			vector = [float(i) for i in vector]
  			embeddings[num_special_symbols + lines_read] = vector
  			lines_read += 1
  			if num_special_symbols + lines_read == limit:
  				break

  	return tf.convert_to_tensor(embeddings, dtype=dtype)



  else:
  	raise IOError("Embedding file location %s not found" % embedding_file)




def get_word_embeddings(enc_inputs, num_enc_symbols, embed_size, embed_algorithm=None, train_embeddings=True, dtype=None):

  #No unsupervised learning algorithm
  if embed_algorithm is None:
    print("***Note - No unsupervised embedding algorithm is present. Word embeddings will be randomized and trained by backpropagation.")
    #create embeddings - this will eventually be moved elsewhere when the embeddings are no longer trained

    with variable_scope.variable_scope("embeddings") as scope:
      emb = tf.get_variable("encoder_embeddings",
                            shape=[num_enc_symbols, embed_size],
                            initializer=tf.random_uniform_initializer(-1.0, 1.0),
                            trainable=True,#have to be trainable because embed algorithm is none.
                            dtype=dtype)

      #get the embedded inputs from the lookup table
      #these will be trained by backpropagation
      embedded_encoder_inputs = tf.nn.embedding_lookup(emb, enc_inputs)

      #these embedded inputs came in as a list. now they will be of the shape (bucket_size, batch_size, embed_size), but we need
      # them to be a sequence input for the lstm layers, so we reshape them back into a list of length bucket_size containing tensors of shape (batch_size, embed_size)
      embedded_encoder_inputs = tf.unstack(embedded_encoder_inputs)
    return embedded_encoder_inputs

  elif embed_algorithm == 'glove':
    print("***Note - Word embeddings will be initialized by glove.")

    if train_embeddings:
      print("***Note - Word embeddings will be trained by backprop")
    else:
      print("***Note - Word embeddings will NOT be trained by backprop")

    with variable_scope.variable_scope("glove_embeddings") as scope:
      emb = tf.get_variable("encoder_embeddings",
                            initializer=initialize_glove_embeddings_tensor(num_enc_symbols, embed_size, dtype=dtype),
                            trainable=train_embeddings,
                            dtype=dtype)

      #get the embedded inputs from the lookup table
      #these will be trained by backpropagation
      embedded_encoder_inputs = tf.nn.embedding_lookup(emb, enc_inputs)

      #these embedded inputs came in as a list. now they will be of the shape (bucket_size, batch_size, embed_size), but we need
      # them to be a sequence input for the lstm layers, so we reshape them back into a list of length bucket_size containing tensors of shape (batch_size, embed_size)
      embedded_encoder_inputs = tf.unstack(embedded_encoder_inputs)
    return embedded_encoder_inputs